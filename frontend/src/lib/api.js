/**
 * api.js
 * -------
 * Lightweight API client for the Adversarial Robustness Driving Planner backend.
 */

async function handleResponse(res) {
  if (!res.ok) {
    let detail = `Request failed (HTTP ${res.status})`;
    try {
      const body = await res.json();
      if (body?.detail) {
        detail = body.detail;
      }
    } catch {
      /* non-JSON error body */
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function fetchAttackOptions() {
  const res = await fetch("/api/attacks");
  return handleResponse(res);
}

export async function fetchSampleData() {
  const res = await fetch("/api/sample-data");
  return handleResponse(res);
}

export async function fetchSampleCameraFile() {
  const res = await fetch("/api/sample-camera");
  if (!res.ok) throw new Error("Could not fetch sample camera image");
  const blob = await res.blob();
  return new File([blob], "camera_same_scene.jpg", { type: "image/jpeg" });
}

export async function fetchSampleRadarFile() {
  const res = await fetch("/api/sample-radar");
  if (!res.ok) throw new Error("Could not fetch sample radar file");
  const blob = await res.blob();
  return new File([blob], "radar_same_scene.pcd", { type: "application/octet-stream" });
}

export async function runEvaluation({
  instruction,
  cameraImage,
  radarFile,
  promptAttack,
  cameraAttack,
  radarAttack,
}) {
  const form = new FormData();

  form.append("instruction", instruction);

  if (cameraImage) {
    form.append("camera_image", cameraImage);
  }
  if (radarFile) {
    form.append("radar_file", radarFile);
  }
  if (promptAttack && promptAttack !== "none") {
    form.append("prompt_attack_name", promptAttack);
  }
  if (cameraAttack && cameraAttack !== "none") {
    form.append("camera_attack_name", cameraAttack);
  }
  if (radarAttack && radarAttack !== "none") {
    form.append("radar_attack_name", radarAttack);
  }

  const res = await fetch("/api/evaluate", {
    method: "POST",
    body: form,
  });

  return handleResponse(res);
}
