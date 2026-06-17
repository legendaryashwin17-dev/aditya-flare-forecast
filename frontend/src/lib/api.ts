const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export interface ForecastSummary {
  max_prob_15min: number;
  max_prob_30min: number;
  max_prob_60min: number;
  alert_level: string;
 flare_risk: string;
}

export interface ForecastResponse {
  status: string;
  source: string;
  solexs_points: number;
  hel1os_points: number;
  model_used: boolean;
  summary: ForecastSummary;
  predictions: Array<{
    timestamp: string;
    prob_15min: number;
    prob_30min: number;
    prob_60min: number;
  }>;
  light_curve?: {
    timestamps: string[];
    solexs_flux: number[];
    hel1os_flux: number[];
  };
  flare_catalogue?: Array<{
    peak_time: string;
    class: string;
    flux: number;
  }>;
}

export interface ModelInfo {
  architecture: string;
  total_params: number;
  solexs_features: number;
  hel1os_features: number;
  input_shape: number[];
  output_shape: number[];
  horizons_minutes: number[];
  training: Record<string, unknown>;
  test_metrics: Record<string, unknown>;
}

export interface StatusResponse {
  status: string;
  model_available: boolean;
  model_path: string;
  architecture: string;
  horizons: number[];
}

export async function fetchStatus(): Promise<StatusResponse> {
  const res = await fetch(`${API_BASE}/api/status`);
  if (!res.ok) throw new Error("Failed to fetch status");
  return res.json();
}

export async function fetchModelInfo(): Promise<ModelInfo> {
  const res = await fetch(`${API_BASE}/api/model-info`);
  if (!res.ok) throw new Error("Failed to fetch model info");
  return res.json();
}

export async function fetchSimulatedForecast(
  durationHours = 24,
  nFlares = 5,
  seed = 42
): Promise<ForecastResponse> {
  const res = await fetch(
    `${API_BASE}/api/forecast/simulated?duration_hours=${durationHours}&n_flares=${nFlares}&seed=${seed}`,
    { method: "POST" }
  );
  if (!res.ok) throw new Error("Failed to fetch simulated forecast");
  return res.json();
}

export async function uploadFitsForecast(
  solexsFile?: File,
  hel1osFile?: File
): Promise<ForecastResponse> {
  const formData = new FormData();
  if (solexsFile) formData.append("solexs_file", solexsFile);
  if (hel1osFile) formData.append("hel1os_file", hel1osFile);

  const res = await fetch(`${API_BASE}/api/forecast`, {
    method: "POST",
    body: formData,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Upload failed" }));
    throw new Error(err.detail || "Failed to upload FITS");
  }
  return res.json();
}
