from __future__ import annotations

import json
import re
import time
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import autocast

from .bootstrap import ensure_repo_imports
from .image_utils import letterbox_rgb_tensor
from .prompt_bank import LoadedPromptPackage, PromptBank

ensure_repo_imports()

from human_spline_localizer.config import load_config as load_localizer_config  # noqa: E402
from human_spline_localizer.model import GlobalSplineLocalizer  # noqa: E402
from human_to_robot_local_spline_translator.config import load_config as load_translator_config  # noqa: E402
from human_to_robot_local_spline_translator.model import LocalHumanToRobotSplineModel  # noqa: E402
from lehome_robot_sim_embedding.config import load_config as load_robot_embedder_config  # noqa: E402
from lehome_robot_sim_embedding.model import RobotSimMultiViewVAE  # noqa: E402


def _as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def _canonical_token(value: str) -> str:
    token = str(value).strip().lower()
    token = token.replace("-", "_").replace(" ", "_")
    token = re.sub(r"[^a-z0-9_]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    return token


def _float_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=np.float32)).to(device=device)


def _bool_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(array, dtype=bool)).to(device=device)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_state_norm(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    return {
        "state_dims": np.asarray(payload["state_dims"], dtype=np.int64),
        "state_mean": np.asarray(payload["state_mean"], dtype=np.float32),
        "state_std": np.clip(np.asarray(payload["state_std"], dtype=np.float32), 1.0e-6, None),
    }


@dataclass(frozen=True)
class PromptRuntimeTensors:
    localizer_coefficients: torch.Tensor
    localizer_left_support: torch.Tensor
    localizer_right_support: torch.Tensor
    localizer_support_midpoint: torch.Tensor
    localizer_support_width: torch.Tensor
    localizer_greville_phase: torch.Tensor
    localizer_basis_200: torch.Tensor
    localizer_mask: torch.Tensor
    translator_global_coefficients: torch.Tensor
    translator_global_knots: torch.Tensor
    translator_global_coeff_counts: torch.Tensor
    translator_global_knot_counts: torch.Tensor


@dataclass
class RuntimeSession:
    session_id: str
    prompt: LoadedPromptPackage
    requested_category_id: str | None
    history_embeddings: deque[np.ndarray]
    history_states: deque[np.ndarray]
    created_time: float
    last_access_time: float
    last_policy_call_index: int | None = None
    last_valid_prediction: dict[str, np.ndarray] | None = None


class RobotEmbedderRuntime:
    def __init__(self, config_path: Path, checkpoint_path: Path, device: torch.device, amp: bool) -> None:
        config = load_robot_embedder_config(config_path, [])
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = RobotSimMultiViewVAE(config)
        model.load_state_dict(payload["model"])
        self.model = model.to(device).eval()
        self.device = device
        self.amp = bool(amp) and device.type == "cuda"
        self.image_size = int(config["model"]["dino"]["image_size"])

    def encode(self, top_rgb: np.ndarray, left_rgb: np.ndarray, right_rgb: np.ndarray) -> np.ndarray:
        images = torch.stack(
            [
                letterbox_rgb_tensor(top_rgb, image_size=self.image_size),
                letterbox_rgb_tensor(left_rgb, image_size=self.image_size),
                letterbox_rgb_tensor(right_rgb, image_size=self.image_size),
            ],
            dim=0,
        ).unsqueeze(0).to(self.device, non_blocking=True)
        with torch.inference_mode():
            with autocast(device_type=self.device.type, enabled=self.amp, dtype=torch.bfloat16):
                dino = self.model.encode_dino(images.flatten(0, 1))
                views = self.model.split_multiview_features(dino, (1, 1))
                posterior = self.model.posterior_from_views(
                    views.top.patches[:, 0],
                    views.left.patches[:, 0],
                    views.right.patches[:, 0],
                )
        return posterior.mean.flatten(1)[0].float().cpu().numpy().astype(np.float32, copy=False)


class HumanLocalizerRuntime:
    def __init__(self, config_path: Path, checkpoint_path: Path, device: torch.device, amp: bool) -> None:
        self.config = load_localizer_config(config_path, [])
        output_root = _as_path(self.config["paths"]["output_root"])
        preprocessing_cfg = self.config["preprocessing"]
        state_norm_path = output_root / str(preprocessing_cfg["state_norm_filename"])
        if not state_norm_path.is_file():
            raise FileNotFoundError(f"Localizer state normalization not found: {state_norm_path}")

        self.state_norm = _load_state_norm(state_norm_path)
        self.history_length = int(self.config["data"]["history_length"])
        interval_cfg = (((self.config.get("model") or {}).get("auxiliary") or {}).get("interval_prediction") or {})
        self.interval_prediction_enabled = bool(interval_cfg.get("enabled", False))

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = GlobalSplineLocalizer(
            config=self.config,
            state_dim=int(self.state_norm["state_dims"].shape[0]),
        )
        model.load_state_dict(payload["model"])
        self.model = model.to(device).eval()
        self.device = device
        self.amp = bool(amp) and device.type == "cuda"


class TranslatorRuntime:
    def __init__(self, config_path: Path, checkpoint_path: Path, device: torch.device, amp: bool) -> None:
        self.config = load_translator_config(config_path, [])
        output_root = _as_path(self.config["paths"]["output_root"])
        preprocessing_cfg = self.config["preprocessing"]
        state_norm_path = output_root / str(preprocessing_cfg["state_norm_filename"])
        if not state_norm_path.is_file():
            raise FileNotFoundError(f"Translator state normalization not found: {state_norm_path}")

        self.state_norm = _load_state_norm(state_norm_path)
        self.history_length = int(self.config["data"]["history_length"])
        self.predicted_u_alpha = float((((self.config.get("inference") or {}).get("predicted_u_alpha")) or 1.0))
        self.teacher_forcing_alpha = float((((self.config.get("inference") or {}).get("teacher_forcing_alpha")) or 0.0))
        self.compressor_gradient_gamma = float((((self.config.get("inference") or {}).get("compressor_gradient_gamma")) or 0.0))
        self.control_point_dim = int(self.config["model"]["human"]["coefficient_input_dim"])

        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = LocalHumanToRobotSplineModel(
            config=self.config,
            state_dim=int(self.state_norm["state_dims"].shape[0]),
        )
        model.load_state_dict(payload["model"])
        self.model = model.to(device).eval()
        self.device = device
        self.amp = bool(amp) and device.type == "cuda"
        self.output_control_count = int(model.num_output_spans + model.degree)
        self.output_knot_count = int(self.output_control_count + model.degree + 1)


class SplineRuntime:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        runtime_cfg = config.get("runtime", {})
        server_cfg = config.get("server", {})
        prompt_cfg = config.get("prompt_selection", {})
        self.device = torch.device(str(runtime_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")))
        self.amp = bool(runtime_cfg.get("amp", True)) and self.device.type == "cuda"
        self.end_mode = str((config.get("end_mode") or {}).get("mode", "predicted_width"))
        self.fixed_future_frames = int((config.get("end_mode") or {}).get("fixed_future_frames", 40))
        self.min_interval_u = float((config.get("end_mode") or {}).get("min_interval_u", 1.0e-4))
        self.fallback_to_fixed_future_frames = bool((config.get("end_mode") or {}).get("fallback_to_fixed_future_frames", True))
        self.use_last_valid_fallback = bool(runtime_cfg.get("use_last_valid_fallback", True))
        self.log_every_n_requests = max(1, int(server_cfg.get("log_every_n_requests", 50)))
        self.prompt_mode = str(prompt_cfg.get("mode", "request"))
        self.fixed_prompt_id = prompt_cfg.get("fixed_prompt_id")
        self.fixed_category_id = prompt_cfg.get("fixed_category_id")
        self.category_aliases = self._build_category_alias_map(prompt_cfg.get("category_aliases", {}))
        self._request_count = 0

        if self.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.prompt_bank = PromptBank(config["paths"]["prompt_bank_root"], seed=int(runtime_cfg.get("seed", 2027)))
        self.robot_embedder = RobotEmbedderRuntime(
            config_path=_as_path(config["robot_embedder"]["config_path"]),
            checkpoint_path=_as_path(config["robot_embedder"]["checkpoint_path"]),
            device=self.device,
            amp=self.amp,
        )
        self.localizer = HumanLocalizerRuntime(
            config_path=_as_path(config["localizer"]["config_path"]),
            checkpoint_path=_as_path(config["localizer"]["checkpoint_path"]),
            device=self.device,
            amp=self.amp,
        )
        self.translator = TranslatorRuntime(
            config_path=_as_path(config["translator"]["config_path"]),
            checkpoint_path=_as_path(config["translator"]["checkpoint_path"]),
            device=self.device,
            amp=self.amp,
        )

        if self.end_mode == "predicted_width" and not self.localizer.width_enabled:
            raise ValueError("Spline runtime is configured for predicted_width end mode, but the localizer checkpoint/config has width_prediction disabled.")

        self.history_capacity = max(self.localizer.history_length, self.translator.history_length)
        self.session_cache_size = max(1, int(server_cfg.get("session_cache_size", 256)))
        self.sessions: OrderedDict[str, RuntimeSession] = OrderedDict()
        self._prompt_tensor_cache: dict[str, PromptRuntimeTensors] = {}

    def server_metadata(self) -> dict[str, Any]:
        return {
            "type": "lehome_spline_runtime",
            "device": str(self.device),
            "amp": bool(self.amp),
            "end_mode": self.end_mode,
            "fixed_future_frames": int(self.fixed_future_frames),
            "min_interval_u": float(self.min_interval_u),
            "history_length": int(self.history_capacity),
            "prompt_categories": self.prompt_bank.categories(),
            "prompt_ids": self.prompt_bank.prompt_ids(),
        }

    def infer(self, payload: dict[str, Any]) -> dict[str, Any]:
        total_start = time.perf_counter()
        session = self._get_or_create_session(payload)
        state_raw = np.asarray(payload["observation/state"], dtype=np.float32).reshape(-1)
        top_rgb = np.asarray(payload["observation/top_rgb"], dtype=np.uint8)
        left_rgb = np.asarray(payload["observation/left_rgb"], dtype=np.uint8)
        right_rgb = np.asarray(payload["observation/right_rgb"], dtype=np.uint8)
        policy_call_index = int(payload.get("policy_call_index", -1))

        embedder_start = time.perf_counter()
        embedding = self.robot_embedder.encode(top_rgb, left_rgb, right_rgb)
        embedder_ms = (time.perf_counter() - embedder_start) * 1000.0

        self._append_session_step(session, embedding=embedding, raw_state=state_raw, policy_call_index=policy_call_index)

        localizer_start = time.perf_counter()
        localizer_result = self._run_localizer(session)
        localizer_ms = (time.perf_counter() - localizer_start) * 1000.0

        interval_start = time.perf_counter()
        interval_result = self._resolve_human_interval(session, localizer_result)
        interval_ms = (time.perf_counter() - interval_start) * 1000.0

        translator_ms = 0.0
        prediction_valid = False
        invalid_reason = interval_result["invalid_reason"]
        used_last_valid_fallback = False
        source = "invalid"
        if interval_result["valid"]:
            translator_start = time.perf_counter()
            translator_result = self._run_translator(
                session=session,
                start_u=float(interval_result["start_u"]),
                end_u=float(interval_result["end_u"]),
            )
            translator_ms = (time.perf_counter() - translator_start) * 1000.0
            prediction_valid = bool(translator_result["prediction_valid"])
            invalid_reason = translator_result["invalid_reason"]
        else:
            translator_result = self._empty_prediction()

        if prediction_valid:
            session.last_valid_prediction = {
                "predicted_robot_coefficients": np.asarray(translator_result["predicted_robot_coefficients"], dtype=np.float32),
                "predicted_robot_knots": np.asarray(translator_result["predicted_robot_knots"], dtype=np.float32),
            }
            source = "fresh"
        elif self.use_last_valid_fallback and session.last_valid_prediction is not None:
            translator_result["predicted_robot_coefficients"] = session.last_valid_prediction["predicted_robot_coefficients"]
            translator_result["predicted_robot_knots"] = session.last_valid_prediction["predicted_robot_knots"]
            used_last_valid_fallback = True
            source = "last_valid_fallback"

        self._request_count += 1
        if self._request_count % self.log_every_n_requests == 0:
            print(
                f"[spline-runtime] requests={self._request_count} session={session.session_id} "
                f"prompt={session.prompt.package.prompt_id} source={source} valid={prediction_valid} "
                f"fallback={used_last_valid_fallback}"
            )

        result = {
            "prediction_valid": bool(prediction_valid),
            "used_last_valid_fallback": bool(used_last_valid_fallback),
            "prediction_source": source,
            "invalid_reason": invalid_reason,
            "prompt_id": session.prompt.package.prompt_id,
            "prompt_category_id": session.prompt.package.category_id,
            "requested_category_id": session.requested_category_id,
            "predicted_human_start_u": float(interval_result["start_u"]) if np.isfinite(interval_result["start_u"]) else float("nan"),
            "predicted_human_end_u": float(interval_result["end_u"]) if np.isfinite(interval_result["end_u"]) else float("nan"),
            "predicted_delta_u": float(localizer_result["delta_u_hat"]) if localizer_result["delta_u_hat"] is not None else None,
            "predicted_end_source": str(interval_result["end_source"]),
            "predicted_robot_coefficients": np.asarray(translator_result["predicted_robot_coefficients"], dtype=np.float32),
            "predicted_robot_knots": np.asarray(translator_result["predicted_robot_knots"], dtype=np.float32),
            "projection_condition_proxy": float(translator_result["projection_condition_proxy"]),
            "span_entropy": float(translator_result["span_entropy"]),
            "human_local_coefficient_count": float(translator_result["human_local_coefficient_count"]),
            "runtime_timing": {
                "embedder_ms": embedder_ms,
                "localizer_ms": localizer_ms,
                "interval_ms": interval_ms,
                "translator_ms": translator_ms,
                "total_ms": (time.perf_counter() - total_start) * 1000.0,
            },
            "session_history_length": int(len(session.history_embeddings)),
        }
        return result

    def _build_category_alias_map(self, raw_aliases: dict[str, Any]) -> dict[str, str]:
        alias_map: dict[str, str] = {}
        for canonical, raw_values in raw_aliases.items():
            canonical_norm = _canonical_token(str(canonical))
            alias_map[canonical_norm] = canonical_norm
            if isinstance(raw_values, (list, tuple)):
                for value in raw_values:
                    alias_map[_canonical_token(str(value))] = canonical_norm
            elif raw_values is not None:
                alias_map[_canonical_token(str(raw_values))] = canonical_norm
        for category_id in self.prompt_bank.categories():
            alias_map.setdefault(_canonical_token(category_id), _canonical_token(category_id))
        return alias_map

    def _resolve_request_category(self, payload: dict[str, Any]) -> str | None:
        explicit = payload.get("prompt_category_id")
        if explicit:
            key = _canonical_token(str(explicit))
            return self.category_aliases.get(key, key)

        garment_type = payload.get("garment_type")
        if garment_type:
            key = _canonical_token(str(garment_type))
            mapped = self.category_aliases.get(key)
            if mapped is not None:
                return mapped

        garment_name = payload.get("garment_name")
        if garment_name:
            candidate = _canonical_token(str(garment_name))
            if candidate in self.category_aliases:
                return self.category_aliases[candidate]
            for alias, canonical in self.category_aliases.items():
                if alias and (alias in candidate or candidate in alias):
                    return canonical
        return None

    def _choose_prompt(self, payload: dict[str, Any]) -> tuple[LoadedPromptPackage, str | None]:
        if self.prompt_mode == "fixed_prompt_id":
            if not self.fixed_prompt_id:
                raise ValueError("prompt_selection.mode=fixed_prompt_id requires prompt_selection.fixed_prompt_id")
            return self.prompt_bank.choose(prompt_id=str(self.fixed_prompt_id)), None

        if payload.get("prompt_id"):
            return self.prompt_bank.choose(prompt_id=str(payload["prompt_id"])), self._resolve_request_category(payload)

        if self.prompt_mode == "fixed_category_random":
            if not self.fixed_category_id:
                raise ValueError("prompt_selection.mode=fixed_category_random requires prompt_selection.fixed_category_id")
            category_id = _canonical_token(str(self.fixed_category_id))
            return self.prompt_bank.choose(category_id=category_id), category_id

        requested_category = self._resolve_request_category(payload)
        if requested_category is None:
            raise ValueError(
                "Spline runtime requires either prompt_id, prompt_category_id, garment_type, or garment_name "
                "to select a prompt package."
            )
        return self.prompt_bank.choose(category_id=requested_category), requested_category

    def _get_or_create_session(self, payload: dict[str, Any]) -> RuntimeSession:
        session_id = str(payload.get("session_id") or "").strip()
        if not session_id:
            raise ValueError("Spline runtime request is missing session_id.")
        session_reset = bool(payload.get("session_reset", False))
        existing = self.sessions.get(session_id)
        if existing is not None and not session_reset:
            existing.last_access_time = time.time()
            self.sessions.move_to_end(session_id)
            return existing

        prompt, requested_category = self._choose_prompt(payload)
        session = RuntimeSession(
            session_id=session_id,
            prompt=prompt,
            requested_category_id=requested_category,
            history_embeddings=deque(maxlen=self.history_capacity),
            history_states=deque(maxlen=self.history_capacity),
            created_time=time.time(),
            last_access_time=time.time(),
        )
        self.sessions[session_id] = session
        self.sessions.move_to_end(session_id)
        while len(self.sessions) > self.session_cache_size:
            self.sessions.popitem(last=False)
        return session

    def _append_session_step(self, session: RuntimeSession, *, embedding: np.ndarray, raw_state: np.ndarray, policy_call_index: int) -> None:
        if session.last_policy_call_index is not None and policy_call_index == session.last_policy_call_index:
            session.last_access_time = time.time()
            return
        session.history_embeddings.append(np.asarray(embedding, dtype=np.float32))
        session.history_states.append(np.asarray(raw_state, dtype=np.float32))
        session.last_policy_call_index = policy_call_index
        session.last_access_time = time.time()

    def _normalize_state(self, raw_state: np.ndarray, norm: dict[str, Any]) -> np.ndarray:
        dims = norm["state_dims"]
        if raw_state.shape[0] <= int(dims.max()):
            raise ValueError(f"Incoming state has dim {raw_state.shape[0]} but runtime expects at least {int(dims.max()) + 1}")
        selected = raw_state[dims]
        return ((selected - norm["state_mean"]) / norm["state_std"]).astype(np.float32, copy=False)

    def _build_history_tensors(
        self,
        session: RuntimeSession,
        *,
        history_length: int,
        state_norm: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embeddings = list(session.history_embeddings)[-history_length:]
        raw_states = list(session.history_states)[-history_length:]
        if not embeddings or not raw_states:
            raise RuntimeError("Runtime session has no history yet.")
        embedding_dim = int(embeddings[-1].shape[0])
        state_dim = int(state_norm["state_dims"].shape[0])
        padded_embeddings = np.zeros((history_length, embedding_dim), dtype=np.float32)
        padded_states = np.zeros((history_length, state_dim), dtype=np.float32)
        mask = np.zeros((history_length,), dtype=bool)
        start = history_length - len(embeddings)
        for index, (embedding, raw_state) in enumerate(zip(embeddings, raw_states, strict=True)):
            target = start + index
            padded_embeddings[target] = np.asarray(embedding, dtype=np.float32)
            padded_states[target] = self._normalize_state(np.asarray(raw_state, dtype=np.float32), state_norm)
            mask[target] = True
        return (
            _float_tensor(padded_embeddings[None, ...], self.device),
            _float_tensor(padded_states[None, ...], self.device),
            _bool_tensor(mask[None, ...], self.device),
        )

    def _prompt_tensors(self, prompt: LoadedPromptPackage) -> PromptRuntimeTensors:
        cached = self._prompt_tensor_cache.get(prompt.package.prompt_id)
        if cached is not None:
            return cached
        coeff_count = int(prompt.coefficient_count)
        coeffs = np.asarray(prompt.global_coefficients[:coeff_count], dtype=np.float32)
        knots = np.asarray(prompt.global_knots, dtype=np.float32)
        localizer_mask = np.ones((1, coeff_count), dtype=bool)
        cached = PromptRuntimeTensors(
            localizer_coefficients=_float_tensor(coeffs[None, ...], self.device),
            localizer_left_support=_float_tensor(prompt.left_support[:coeff_count][None, ...], self.device),
            localizer_right_support=_float_tensor(prompt.right_support[:coeff_count][None, ...], self.device),
            localizer_support_midpoint=_float_tensor(prompt.support_midpoint[:coeff_count][None, ...], self.device),
            localizer_support_width=_float_tensor(prompt.support_width[:coeff_count][None, ...], self.device),
            localizer_greville_phase=_float_tensor(prompt.greville_phase[:coeff_count][None, ...], self.device),
            localizer_basis_200=_float_tensor(prompt.basis[:, :coeff_count][None, ...], self.device),
            localizer_mask=_bool_tensor(localizer_mask, self.device),
            translator_global_coefficients=_float_tensor(coeffs[None, ...], self.device),
            translator_global_knots=_float_tensor(knots[None, ...], self.device),
            translator_global_coeff_counts=torch.tensor([coeff_count], dtype=torch.int64, device=self.device),
            translator_global_knot_counts=torch.tensor([int(knots.shape[0])], dtype=torch.int64, device=self.device),
        )
        self._prompt_tensor_cache[prompt.package.prompt_id] = cached
        return cached

    def _run_localizer(self, session: RuntimeSession) -> dict[str, Any]:
        history_embeddings, history_states, history_mask = self._build_history_tensors(
            session,
            history_length=self.localizer.history_length,
            state_norm=self.localizer.state_norm,
        )
        prompt_tensors = self._prompt_tensors(session.prompt)
        with torch.inference_mode():
            with autocast(device_type=self.device.type, enabled=self.localizer.amp, dtype=torch.bfloat16):
                outputs = self.localizer.model(
                    robot_history_embeddings=history_embeddings,
                    robot_history_states=history_states,
                    robot_history_mask=history_mask,
                    human_coefficients=prompt_tensors.localizer_coefficients,
                    human_left_support=prompt_tensors.localizer_left_support,
                    human_right_support=prompt_tensors.localizer_right_support,
                    human_support_midpoint=prompt_tensors.localizer_support_midpoint,
                    human_support_width=prompt_tensors.localizer_support_width,
                    human_greville_phase=prompt_tensors.localizer_greville_phase,
                    human_basis_200=prompt_tensors.localizer_basis_200,
                    human_mask=prompt_tensors.localizer_mask,
                )
        result = {
            "u_hat": float(outputs["u_hat"][0].detach().float().cpu().item()),
            "u_end_hat": None,
            "delta_u_hat": None,
        }
        if "u_end_hat" in outputs:
            result["u_end_hat"] = float(outputs["u_end_hat"][0].detach().float().cpu().item())
        if "delta_u_hat" in outputs:
            result["delta_u_hat"] = float(outputs["delta_u_hat"][0].detach().float().cpu().item())
        return result

    def _resolve_human_interval(self, session: RuntimeSession, localizer_result: dict[str, Any]) -> dict[str, Any]:
        start_u = float(localizer_result["u_hat"])
        invalid_reason = None
        end_u = float("nan")
        end_source = "none"

        if self.end_mode == "predicted_width":
            predicted_end_u = localizer_result.get("u_end_hat")
            if predicted_end_u is None or not np.isfinite(float(predicted_end_u)):
                invalid_reason = "predicted_interval_requested_but_localizer_has_no_valid_end_output"
            else:
                end_u = float(predicted_end_u)
                end_source = "localizer_predicted_interval"

        if (not np.isfinite(end_u)) and self.fallback_to_fixed_future_frames:
            end_u = float(session.prompt.future_end_u_from_frame_offset(start_u, self.fixed_future_frames))
            end_source = "fixed_future_frames_fallback"
            invalid_reason = None

        valid = bool(np.isfinite(start_u) and np.isfinite(end_u) and (end_u - start_u) >= self.min_interval_u)
        if not valid and invalid_reason is None:
            invalid_reason = "degenerate_or_nonfinite_human_interval"
        return {
            "valid": valid,
            "invalid_reason": invalid_reason,
            "start_u": start_u,
            "end_u": end_u,
            "end_source": end_source,
        }

    def _run_translator(self, session: RuntimeSession, *, start_u: float, end_u: float) -> dict[str, Any]:
        history_embeddings, history_states, history_mask = self._build_history_tensors(
            session,
            history_length=self.translator.history_length,
            state_norm=self.translator.state_norm,
        )
        prompt_tensors = self._prompt_tensors(session.prompt)
        with torch.inference_mode():
            with autocast(device_type=self.device.type, enabled=self.translator.amp, dtype=torch.bfloat16):
                outputs = self.translator.model(
                    robot_history_embeddings=history_embeddings,
                    robot_history_states=history_states,
                    robot_history_mask=history_mask,
                    human_global_coefficients=prompt_tensors.translator_global_coefficients,
                    human_global_knots=prompt_tensors.translator_global_knots,
                    human_global_coeff_counts=prompt_tensors.translator_global_coeff_counts,
                    human_global_knot_counts=prompt_tensors.translator_global_knot_counts,
                    human_input_start_u=torch.tensor([start_u], dtype=torch.float32, device=self.device),
                    human_input_end_u=torch.tensor([end_u], dtype=torch.float32, device=self.device),
                    dense_robot_teacher=None,
                    teacher_forcing_alpha=float(self.translator.teacher_forcing_alpha),
                    compressor_gradient_gamma=float(self.translator.compressor_gradient_gamma),
                )

        coeffs = outputs["predicted_coefficients"][0].detach().float().cpu().numpy().astype(np.float32, copy=False)
        knots = outputs["predicted_knots"][0].detach().float().cpu().numpy().astype(np.float32, copy=False)
        finite = (
            np.isfinite(coeffs).all()
            and np.isfinite(knots).all()
            and np.isfinite(float(outputs["projection_condition_proxy"][0].detach().float().cpu().item()))
            and np.isfinite(float(outputs["span_entropy"][0].detach().float().cpu().item()))
        )
        return {
            "prediction_valid": bool(finite),
            "invalid_reason": None if finite else "non_finite_translator_output",
            "predicted_robot_coefficients": coeffs,
            "predicted_robot_knots": knots,
            "projection_condition_proxy": float(outputs["projection_condition_proxy"][0].detach().float().cpu().item()),
            "span_entropy": float(outputs["span_entropy"][0].detach().float().cpu().item()),
            "human_local_coefficient_count": float(outputs["human_local_coefficient_count"][0].detach().float().cpu().item()),
        }

    def _empty_prediction(self) -> dict[str, Any]:
        return {
            "prediction_valid": False,
            "invalid_reason": "invalid_human_interval",
            "predicted_robot_coefficients": np.zeros(
                (self.translator.output_control_count, self.translator.control_point_dim),
                dtype=np.float32,
            ),
            "predicted_robot_knots": np.zeros((self.translator.output_knot_count,), dtype=np.float32),
            "projection_condition_proxy": float("nan"),
            "span_entropy": float("nan"),
            "human_local_coefficient_count": 0.0,
        }
