"use client"

import dynamic from "next/dynamic"
import { useState, useEffect } from "react"
import { Activity, Zap, Sun, Radio, BarChart3, ArrowRight, Download, Upload, Satellite } from "lucide-react"
import { LiquidButton } from "@/components/ui/liquid-glass-button"
import { GooeyText } from "@/components/ui/gooey-text-morphing"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"

const ShaderAnimation = dynamic(
  () => import("@/components/ui/shader-animation").then(m => ({ default: m.ShaderAnimation })),
  { ssr: false }
)

const RadialOrbitalTimeline = dynamic(
  () => import("@/components/ui/radial-orbital-timeline"),
  { ssr: false }
)

const pipelineSteps = [
  { icon: Satellite, label: "SoLEXS", sub: "2-22 keV", color: "#00ffaa" },
  { icon: Radio, label: "HEL1OS", sub: "8-150 keV", color: "#0088ff" },
  { icon: Activity, label: "17 Features", sub: "Physics-informed", color: "#c084fc" },
  { icon: Zap, label: "CNN-BiLSTM", sub: "~102K params", color: "#ffd93d" },
  { icon: BarChart3, label: "Forecast", sub: "15/30/60 min", color: "#ff6b6b" },
]

const timelineData = [
  {
    id: 1,
    title: "Data Ingestion",
    date: "Step 1",
    content: "SoLEXS + HEL1OS FITS from PRADAN portal. 1s native cadence, 86K samples/day.",
    category: "Data",
    icon: Download,
    relatedIds: [2],
    status: "completed" as const,
    energy: 100,
  },
  {
    id: 2,
    title: "Preprocessing",
    date: "Step 2",
    content: "10s binning, 17 physics features, z-score standardization.",
    category: "Preprocessing",
    icon: Activity,
    relatedIds: [1, 3],
    status: "completed" as const,
    energy: 85,
  },
  {
    id: 3,
    title: "8-22 keV Overlap",
    date: "Step 3",
    content: "Spectral hardness ratio + Pearson cross-correlation in precursor band.",
    category: "Features",
    icon: Sun,
    relatedIds: [2, 4],
    status: "completed" as const,
    energy: 70,
  },
  {
    id: 4,
    title: "ParallelCNN-BiLSTM",
    date: "Step 4",
    content: "Dual-branch architecture with overlap convolution and bidirectional LSTM.",
    category: "Model",
    icon: Zap,
    relatedIds: [3, 5],
    status: "in-progress" as const,
    energy: 55,
  },
  {
    id: 5,
    title: "Multi-Head Forecast",
    date: "Step 5",
    content: "3 independent heads: 15min (TSS=0.75), 30min, 60min horizon predictions.",
    category: "Output",
    icon: BarChart3,
    relatedIds: [4],
    status: "in-progress" as const,
    energy: 40,
  },
]

function StatCard({ value, label, color }: { value: string; label: string; color: string }) {
  return (
    <div className="glass-card p-4 text-center">
      <div className="text-2xl font-bold font-mono" style={{ color }}>{value}</div>
      <div className="text-xs text-white/50 mt-1 uppercase tracking-wider">{label}</div>
    </div>
  )
}

export default function Home() {
  const [mounted, setMounted] = useState(false)

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <div className="min-h-screen bg-[#030305] text-white overflow-x-hidden">
      <div className="noise-overlay" />

      {/* Hero Section with Shader Background */}
      <section className="relative h-screen flex items-center justify-center">
        <div className="absolute inset-0 z-0">
          {mounted && <ShaderAnimation />}
        </div>

        <div className="relative z-10 text-center px-4 max-w-5xl mx-auto">
          <div className="animate-fade-slide-up">
            <Badge className="mb-6 px-4 py-1.5 bg-white/5 border-white/10 text-white/70 text-xs tracking-widest uppercase">
              Bharatiya Antariksh Hackathon 2026
            </Badge>
          </div>

          <div className="animate-fade-slide-up delay-100">
            <h1 className="text-5xl md:text-7xl lg:text-8xl font-bold tracking-tighter mb-4">
              <span className="gradient-text">Solar Flare</span>
              <br />
              <span className="text-white">Forecast</span>
            </h1>
          </div>

          <div className="animate-fade-slide-up delay-200 h-20 flex items-center justify-center">
            <GooeyText
              texts={["SoLEXS + HEL1OS", "Aditya-L1", "8-22 keV Overlap", "Nowcast & Forecast"]}
              morphTime={1.5}
              cooldownTime={0.5}
              textClassName="text-xl md:text-2xl font-medium"
            />
          </div>

          <div className="animate-fade-slide-up delay-300 mt-8">
            <p className="text-white/50 max-w-2xl mx-auto text-lg leading-relaxed">
              AI-powered solar flare prediction using combined soft and hard X-ray
              time-series from Aditya-L1. Targeting TSS &ge; 0.65 on &ge;C-class flares.
            </p>
          </div>

          <div className="animate-fade-slide-up delay-400 mt-10 flex flex-col sm:flex-row gap-4 justify-center">
            <a href="/dashboard">
              <LiquidButton size="lg" className="text-base">
                Launch Forecast
                <ArrowRight className="ml-2 h-4 w-4" />
              </LiquidButton>
            </a>
            <Button variant="outline" size="lg" className="border-white/10 text-white/70 hover:bg-white/5">
              View Pipeline
            </Button>
          </div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2 animate-bounce">
          <div className="w-6 h-10 rounded-full border-2 border-white/20 flex items-start justify-center p-1">
            <div className="w-1.5 h-3 bg-white/40 rounded-full animate-pulse" />
          </div>
        </div>
      </section>

      {/* Stats Bar */}
      <section className="relative z-10 -mt-20 px-4">
        <div className="max-w-5xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard value="0.75" label="15min TSS" color="#00ffaa" />
          <StatCard value="0.91" label="15min AUC" color="#0088ff" />
          <StatCard value="102K" label="Parameters" color="#c084fc" />
          <StatCard value="17" label="Features" color="#ffd93d" />
        </div>
      </section>

      {/* Pipeline Visualization */}
      <section className="py-32 px-4">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <div className="text-xs text-white/40 uppercase tracking-[0.2em] mb-3">Architecture</div>
            <h2 className="text-3xl md:text-5xl font-bold tracking-tight">
              Processing <span className="gradient-text">Pipeline</span>
            </h2>
          </div>

          {/* Pipeline Steps */}
          <div className="flex flex-wrap justify-center gap-4 mb-20">
            {pipelineSteps.map((step, i) => (
              <div key={i} className="pipeline-connector flex items-center gap-3 glass-card px-5 py-3">
                <step.icon size={20} style={{ color: step.color }} />
                <div>
                  <div className="text-sm font-semibold text-white">{step.label}</div>
                  <div className="text-xs text-white/40">{step.sub}</div>
                </div>
              </div>
            ))}
          </div>

          {/* Radial Orbital Timeline */}
          <div className="h-[600px] rounded-2xl overflow-hidden border border-white/5">
            <RadialOrbitalTimeline timelineData={timelineData} />
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-20 px-4">
        <div className="max-w-5xl mx-auto">
          <div className="text-center mb-16">
            <div className="text-xs text-white/40 uppercase tracking-[0.2em] mb-3">Innovation</div>
            <h2 className="text-3xl md:text-5xl font-bold tracking-tight">
              Key <span className="gradient-text">Features</span>
            </h2>
          </div>

          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                title: "8-22 keV Overlap Band",
                desc: "Pre-flare precursor brightening detection using correlated emission across SoLEXS and HEL1OS instruments.",
                color: "#00ffaa",
              },
              {
                title: "Parallel Dual-Branch",
                desc: "Sensor-specific CNN branches preserve noise distributions before overlap convolution fusion.",
                color: "#0088ff",
              },
              {
                title: "Focal Loss Training",
                desc: "Alpha=0.75, gamma=2.0 handles severe class imbalance in solar flare datasets.",
                color: "#c084fc",
              },
              {
                title: "Multi-Head Forecast",
                desc: "3 independent prediction heads for 15, 30, and 60 minute horizons.",
                color: "#ffd93d",
              },
              {
                title: "Physics-Informed Features",
                desc: "Spectral hardness ratio, rolling Pearson correlation, background-subtracted fluxes.",
                color: "#ff6b6b",
              },
              {
                title: "Transfer Learning Ready",
                desc: "Pre-train on 20 years of GOES XRS data, fine-tune on Aditya-L1.",
                color: "#00ffaa",
              },
            ].map((feat, i) => (
              <Card key={i} className="glass-card border-white/5 bg-transparent hover:border-white/10 transition-all group">
                <CardHeader>
                  <div className="w-2 h-2 rounded-full mb-3" style={{ background: feat.color }} />
                  <CardTitle className="text-lg text-white group-hover:text-white/90">{feat.title}</CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-white/40 leading-relaxed">{feat.desc}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-4 border-t border-white/5">
        <div className="max-w-5xl mx-auto text-center">
          <div className="text-xs text-white/30 uppercase tracking-[0.2em]">
            Bharatiya Antariksh Hackathon 2026 &middot; Problem Statement 15
          </div>
          <div className="mt-2 text-xs text-white/20">
            Aditya-L1 SoLEXS + HEL1OS &middot; Parallel CNN-BiLSTM &middot; TSS &ge; 0.65 Target
          </div>
        </div>
      </footer>
    </div>
  )
}
