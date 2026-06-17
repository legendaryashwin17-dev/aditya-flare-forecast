"use client"

import { useState, useEffect } from "react"
import { Activity, Zap, Sun, Radio, BarChart3, ArrowRight, RefreshCw, Upload, AlertTriangle, CheckCircle } from "lucide-react"
import { LiquidButton } from "@/components/ui/liquid-glass-button"
import { GooeyText } from "@/components/ui/gooey-text-morphing"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  fetchSimulatedForecast,
  uploadFitsForecast,
  type ForecastResponse,
  type StatusResponse,
  fetchStatus,
} from "@/lib/api"

function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.round(value * 100)
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-white/50">{label}</span>
        <span className="font-mono" style={{ color }}>{pct}%</span>
      </div>
      <div className="h-2 bg-white/5 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-700"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
    </div>
  )
}

function MiniChart({ data, color, label }: { data: number[]; color: string; label: string }) {
  if (!data.length) return null
  const max = Math.max(...data)
  const min = Math.min(...data)
  const range = max - min || 1
  const w = 200
  const h = 40
  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * w
    const y = h - ((v - min) / range) * h
    return `${x},${y}`
  }).join(" ")

  return (
    <div>
      <div className="text-xs text-white/40 mb-1">{label}</div>
      <svg viewBox={`0 0 ${w} ${h}`} className="w-full h-10">
        <polyline fill="none" stroke={color} strokeWidth="1.5" points={points} opacity="0.8" />
      </svg>
    </div>
  )
}

export default function DashboardPage() {
  const [status, setStatus] = useState<StatusResponse | null>(null)
  const [forecast, setForecast] = useState<ForecastResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchStatus().then(setStatus).catch(() => {})
    loadSimulated()
  }, [])

  async function loadSimulated() {
    setLoading(true)
    setError(null)
    try {
      const data = await fetchSimulatedForecast(24, 5, 42)
      setForecast(data)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>, type: "solexs" | "hel1os") {
    const file = e.target.files?.[0]
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const data = type === "solexs"
        ? await uploadFitsForecast(file, undefined)
        : await uploadFitsForecast(undefined, file)
      setForecast(data)
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const summary = forecast?.summary
  const alertLevel = summary?.alert_level || "LOW"
  const alertColor = alertLevel === "HIGH" ? "#ff4444" : alertLevel === "MODERATE" ? "#ffd93d" : "#00ffaa"

  return (
    <div className="min-h-screen bg-[#030305] text-white">
      <div className="noise-overlay" />

      {/* Header */}
      <header className="sticky top-0 z-50 border-b border-white/5 bg-[#030305]/80 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-emerald-500 to-blue-500 flex items-center justify-center">
              <Sun size={16} className="text-white" />
            </div>
            <div>
              <div className="text-sm font-bold">Aditya-L1</div>
              <div className="text-[10px] text-white/40 uppercase tracking-widest">Flare Forecast</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Badge className="bg-white/5 border-white/10 text-white/50 text-[10px]">
              {status?.model_available ? "Model Loaded" : "Simulated"}
            </Badge>
            <Button
              variant="ghost"
              size="sm"
              onClick={loadSimulated}
              disabled={loading}
              className="text-white/50 hover:text-white"
            >
              <RefreshCw size={14} className={loading ? "animate-spin" : ""} />
            </Button>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-8 space-y-8">
        {/* Alert Banner */}
        <div
          className="glass-card p-4 flex items-center gap-4"
          style={{ borderColor: `${alertColor}20` }}
        >
          {alertLevel !== "LOW" ? (
            <AlertTriangle size={20} style={{ color: alertColor }} />
          ) : (
            <CheckCircle size={20} style={{ color: alertColor }} />
          )}
          <div className="flex-1">
            <div className="text-sm font-semibold" style={{ color: alertColor }}>
              {alertLevel} Alert — {summary?.flare_risk || "Low Risk"}
            </div>
            <div className="text-xs text-white/40">
              Max probabilities: 15min {Math.round((summary?.max_prob_15min || 0) * 100)}% · 
              30min {Math.round((summary?.max_prob_30min || 0) * 100)}% · 
              60min {Math.round((summary?.max_prob_60min || 0) * 100)}%
            </div>
          </div>
          <Badge
            className="text-xs font-mono"
            style={{ background: `${alertColor}15`, color: alertColor, border: `1px solid ${alertColor}30` }}
          >
            {alertLevel}
          </Badge>
        </div>

        {/* Upload Section */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Card className="glass-card border-white/5 bg-transparent">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Upload size={14} className="text-emerald-400" />
                SoLEXS FITS
              </CardTitle>
            </CardHeader>
            <CardContent>
              <label className="block cursor-pointer">
                <div className="border border-dashed border-white/10 rounded-lg p-6 text-center hover:border-emerald-500/30 transition-colors">
                  <input
                    type="file"
                    accept=".fits,.fit,.lc"
                    className="hidden"
                    onChange={(e) => handleUpload(e, "solexs")}
                  />
                  <div className="text-xs text-white/40">
                    Drop .lc file or click to upload
                  </div>
                </div>
              </label>
            </CardContent>
          </Card>

          <Card className="glass-card border-white/5 bg-transparent">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Upload size={14} className="text-blue-400" />
                HEL1OS FITS
              </CardTitle>
            </CardHeader>
            <CardContent>
              <label className="block cursor-pointer">
                <div className="border border-dashed border-white/10 rounded-lg p-6 text-center hover:border-blue-500/30 transition-colors">
                  <input
                    type="file"
                    accept=".fits,.fit"
                    className="hidden"
                    onChange={(e) => handleUpload(e, "hel1os")}
                  />
                  <div className="text-xs text-white/40">
                    Drop lightcurve_*.fits or click to upload
                  </div>
                </div>
              </label>
            </CardContent>
          </Card>
        </div>

        {error && (
          <div className="glass-card p-3 border-red-500/20 text-red-400 text-xs">{error}</div>
        )}

        {/* Forecast Probabilities */}
        {forecast && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Card className="glass-card border-white/5 bg-transparent">
              <CardHeader className="pb-2">
                <CardTitle className="text-xs text-white/40 flex items-center gap-2">
                  <Zap size={12} className="text-emerald-400" />
                  15-Minute Horizon
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <ProbBar label="Flare Probability" value={summary?.max_prob_15min || 0} color="#00ffaa" />
                <MiniChart
                  data={forecast.predictions.map(p => p.prob_15min)}
                  color="#00ffaa"
                  label="Time series"
                />
              </CardContent>
            </Card>

            <Card className="glass-card border-white/5 bg-transparent">
              <CardHeader className="pb-2">
                <CardTitle className="text-xs text-white/40 flex items-center gap-2">
                  <Activity size={12} className="text-blue-400" />
                  30-Minute Horizon
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <ProbBar label="Flare Probability" value={summary?.max_prob_30min || 0} color="#0088ff" />
                <MiniChart
                  data={forecast.predictions.map(p => p.prob_30min)}
                  color="#0088ff"
                  label="Time series"
                />
              </CardContent>
            </Card>

            <Card className="glass-card border-white/5 bg-transparent">
              <CardHeader className="pb-2">
                <CardTitle className="text-xs text-white/40 flex items-center gap-2">
                  <BarChart3 size={12} className="text-purple-400" />
                  60-Minute Horizon
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <ProbBar label="Flare Probability" value={summary?.max_prob_60min || 0} color="#c084fc" />
                <MiniChart
                  data={forecast.predictions.map(p => p.prob_60min)}
                  color="#c084fc"
                  label="Time series"
                />
              </CardContent>
            </Card>
          </div>
        )}

        {/* Light Curves */}
        {forecast?.light_curve && (
          <Card className="glass-card border-white/5 bg-transparent">
            <CardHeader>
              <CardTitle className="text-sm flex items-center gap-2">
                <Radio size={14} className="text-emerald-400" />
                Light Curves
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <MiniChart data={forecast.light_curve.solexs_flux} color="#00ffaa" label="SoLEXS (2-22 keV)" />
              <MiniChart data={forecast.light_curve.hel1os_flux} color="#0088ff" label="HEL1OS (8-150 keV)" />
            </CardContent>
          </Card>
        )}

        {/* Flare Catalogue */}
        {forecast?.flare_catalogue && forecast.flare_catalogue.length > 0 && (
          <Card className="glass-card border-white/5 bg-transparent">
            <CardHeader>
              <CardTitle className="text-sm">Flare Catalogue</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {forecast.flare_catalogue.map((flare, i) => (
                  <div key={i} className="flex items-center justify-between py-2 border-b border-white/5 last:border-0">
                    <div className="flex items-center gap-3">
                      <div className="w-2 h-2 rounded-full bg-red-500" />
                      <span className="text-xs font-mono text-white/60">{flare.peak_time}</span>
                    </div>
                    <Badge className="bg-red-500/10 text-red-400 border-red-500/20 text-[10px]">
                      {flare.class}-class
                    </Badge>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}

        {/* Model Info */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { label: "Architecture", value: "CNN-BiLSTM", icon: Zap, color: "#00ffaa" },
            { label: "Parameters", value: "102K", icon: Activity, color: "#0088ff" },
            { label: "Features", value: "17", icon: Sun, color: "#c084fc" },
            { label: "Horizons", value: "3", icon: BarChart3, color: "#ffd93d" },
          ].map((item, i) => (
            <div key={i} className="glass-card p-3 text-center">
              <item.icon size={16} className="mx-auto mb-1" style={{ color: item.color }} />
              <div className="text-lg font-bold font-mono" style={{ color: item.color }}>{item.value}</div>
              <div className="text-[10px] text-white/40 uppercase tracking-wider">{item.label}</div>
            </div>
          ))}
        </div>
      </main>
    </div>
  )
}
