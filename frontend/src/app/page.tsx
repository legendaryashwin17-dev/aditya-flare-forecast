"use client"

import dynamic from "next/dynamic"
import { useState, useEffect, useRef } from "react"
import { motion, useInView, useScroll, useTransform } from "framer-motion"
import { Activity, Zap, Sun, Radio, BarChart3, ArrowRight, Download, Upload, Satellite, AlertTriangle } from "lucide-react"
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
    content: "3 independent heads: 15min, 30min, 60min horizon predictions.",
    category: "Output",
    icon: BarChart3,
    relatedIds: [4],
    status: "in-progress" as const,
    energy: 40,
  },
]

function AnimatedCard({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true, margin: "-80px" })

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 40, filter: "blur(8px)" }}
      animate={isInView ? { opacity: 1, y: 0, filter: "blur(0px)" } : {}}
      transition={{ duration: 0.7, delay, ease: [0.16, 1, 0.3, 1] }}
    >
      {children}
    </motion.div>
  )
}

function StatCard({ value, label, color, desc }: { value: string; label: string; color: string; desc?: string }) {
  const ref = useRef(null)
  const isInView = useInView(ref, { once: true })

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, scale: 0.9 }}
      animate={isInView ? { opacity: 1, scale: 1 } : {}}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="glass-card p-5 group hover:border-white/10 transition-all duration-500 relative overflow-hidden"
    >
      <div
        className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500"
        style={{ background: `radial-gradient(circle at 50% 50%, ${color}08, transparent 70%)` }}
      />
      <div className="relative">
        <div className="text-3xl font-bold font-mono tracking-tight" style={{ color }}>{value}</div>
        <div className="text-xs text-white/50 mt-1.5 uppercase tracking-wider font-medium">{label}</div>
        {desc && <div className="text-[10px] text-white/30 mt-1">{desc}</div>}
      </div>
    </motion.div>
  )
}

function GlowLine() {
  return (
    <div className="h-px w-full bg-gradient-to-r from-transparent via-[#00ffaa]/30 to-transparent" />
  )
}

export default function Home() {
  const [mounted, setMounted] = useState(false)
  const heroRef = useRef(null)
  const { scrollYProgress } = useScroll({
    target: heroRef,
    offset: ["start start", "end start"],
  })
  const heroOpacity = useTransform(scrollYProgress, [0, 0.5], [1, 0])
  const heroScale = useTransform(scrollYProgress, [0, 0.5], [1, 0.95])
  const heroBlur = useTransform(scrollYProgress, [0, 0.5], [0, 8])

  useEffect(() => {
    setMounted(true)
  }, [])

  return (
    <div className="min-h-screen bg-[#030305] text-white overflow-x-hidden">
      <div className="noise-overlay" />

      {/* Hero — LEFT-ALIGNED per design-taste-frontend */}
      <motion.section
        ref={heroRef}
        style={{ opacity: heroOpacity, scale: heroScale }}
        className="relative min-h-screen flex items-center"
      >
        <div className="absolute inset-0 z-0">
          {mounted && <ShaderAnimation />}
        </div>

        <div className="relative z-10 px-6 md:px-12 lg:px-20 max-w-6xl">
          <motion.div
            initial={{ opacity: 0, x: -30 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
          >
            <Badge className="mb-6 px-4 py-1.5 bg-white/5 border-white/10 text-white/70 text-xs tracking-widest uppercase">
              Bharatiya Antariksh Hackathon 2026
            </Badge>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, x: -40 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.1, ease: [0.16, 1, 0.3, 1] }}
            className="text-5xl md:text-7xl lg:text-8xl font-bold tracking-tighter mb-4 text-left"
          >
            <span className="gradient-text">Solar Flare</span>
            <br />
            <span className="text-white">Forecast</span>
          </motion.h1>

          <motion.div
            initial={{ opacity: 0, x: -40 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.2, ease: [0.16, 1, 0.3, 1] }}
            className="h-16 flex items-center"
          >
            <GooeyText
              texts={["SoLEXS + HEL1OS", "Aditya-L1", "8-22 keV Overlap", "Nowcast & Forecast"]}
              morphTime={1.5}
              cooldownTime={0.5}
              textClassName="text-xl md:text-2xl font-medium"
            />
          </motion.div>

          <motion.p
            initial={{ opacity: 0, x: -40 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.8, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="text-white/50 max-w-xl text-lg leading-relaxed text-left mt-4"
          >
            AI-powered solar flare prediction using combined soft and hard X-ray
            time-series from India&apos;s first solar observatory.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4, ease: [0.16, 1, 0.3, 1] }}
            className="mt-8 flex flex-col sm:flex-row gap-4"
          >
            <a href="/dashboard">
              <LiquidButton size="lg" className="text-base">
                Launch Forecast
                <ArrowRight className="ml-2 h-4 w-4" />
              </LiquidButton>
            </a>
            <Button variant="outline" size="lg" className="border-white/10 text-white/70 hover:bg-white/5">
              View Pipeline
            </Button>
          </motion.div>
        </div>

        {/* Scroll indicator */}
        <div className="absolute bottom-8 left-1/2 -translate-x-1/2">
          <motion.div
            animate={{ y: [0, 8, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
            className="w-6 h-10 rounded-full border-2 border-white/20 flex items-start justify-center p-1"
          >
            <div className="w-1.5 h-3 bg-white/40 rounded-full" />
          </motion.div>
        </div>
      </motion.section>

      {/* Stats Bar — honest data */}
      <section className="relative z-10 -mt-20 px-6">
        <div className="max-w-6xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard value="~102K" label="Parameters" color="#c084fc" desc="Lightweight model" />
          <StatCard value="17" label="Features" color="#ffd93d" desc="Physics-informed" />
          <StatCard value="3" label="Horizons" color="#ff6b6b" desc="15 / 30 / 60 min" />
          <StatCard value="2" label="Instruments" color="#0088ff" desc="SoLEXS + HEL1OS" />
        </div>
      </section>

      {/* Honest Disclaimer */}
      <section className="py-12 px-6">
        <div className="max-w-3xl mx-auto">
          <AnimatedCard>
            <div className="glass-card p-6 border-[#ffa502]/20">
              <div className="flex items-start gap-3">
                <AlertTriangle size={18} className="text-[#ffa502] mt-0.5 flex-shrink-0" />
                <div>
                  <div className="text-sm font-semibold text-[#ffa502] mb-1">Proof of Concept</div>
                  <p className="text-xs text-white/40 leading-relaxed">
                    Current model trained on ~12 hours of PRADAN data (June 14-15, 2026).
                    SoLEXS and HEL1OS observations do not yet overlap temporally.
                    Transfer learning from GOES XRS is planned but not yet implemented.
                    Performance metrics shown are from demo/simulated data.
                  </p>
                </div>
              </div>
            </div>
          </AnimatedCard>
        </div>
      </section>

      {/* Pipeline Visualization */}
      <section className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <AnimatedCard>
            <div className="text-left mb-16">
              <div className="text-xs text-white/40 uppercase tracking-[0.2em] mb-3">Architecture</div>
              <h2 className="text-3xl md:text-5xl font-bold tracking-tight">
                Processing <span className="gradient-text">Pipeline</span>
              </h2>
            </div>
          </AnimatedCard>

          {/* Pipeline Steps — horizontal scroll on mobile */}
          <AnimatedCard delay={0.1}>
            <div className="flex flex-wrap justify-start gap-4 mb-20">
              {pipelineSteps.map((step, i) => (
                <motion.div
                  key={i}
                  whileHover={{ scale: 1.05, y: -2 }}
                  className="pipeline-connector flex items-center gap-3 glass-card px-5 py-3 cursor-default"
                >
                  <step.icon size={20} style={{ color: step.color }} />
                  <div>
                    <div className="text-sm font-semibold text-white">{step.label}</div>
                    <div className="text-xs text-white/40">{step.sub}</div>
                  </div>
                </motion.div>
              ))}
            </div>
          </AnimatedCard>

          {/* Radial Orbital Timeline */}
          <AnimatedCard delay={0.2}>
            <div className="h-[600px] rounded-2xl overflow-hidden border border-white/5">
              <RadialOrbitalTimeline timelineData={timelineData} />
            </div>
          </AnimatedCard>
        </div>
      </section>

      <GlowLine />

      {/* Features Grid — interactive hover */}
      <section className="py-20 px-6">
        <div className="max-w-6xl mx-auto">
          <AnimatedCard>
            <div className="text-left mb-16">
              <div className="text-xs text-white/40 uppercase tracking-[0.2em] mb-3">Innovation</div>
              <h2 className="text-3xl md:text-5xl font-bold tracking-tight">
                Key <span className="gradient-text">Features</span>
              </h2>
            </div>
          </AnimatedCard>

          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                title: "8-22 keV Overlap Band",
                desc: "Spectral hardness ratio and Pearson cross-correlation in the energy band where SoLEXS and HEL1OS overlap.",
                color: "#00ffaa",
                note: "Requires simultaneous data from both instruments",
              },
              {
                title: "Parallel Dual-Branch",
                desc: "Sensor-specific CNN branches preserve noise distributions before overlap convolution fusion.",
                color: "#0088ff",
                note: "1D-Conv(32, k=3) per instrument",
              },
              {
                title: "Focal Loss Training",
                desc: "Alpha=0.75, gamma=2.0 handles severe class imbalance in solar flare datasets.",
                color: "#c084fc",
                note: "Lin et al. 2017",
              },
              {
                title: "Multi-Head Forecast",
                desc: "3 independent prediction heads for 15, 30, and 60 minute horizons.",
                color: "#ffd93d",
                note: "Independent Dense(32)->Dense(1, sigmoid)",
              },
              {
                title: "Physics-Informed Features",
                desc: "Spectral hardness ratio, rolling Pearson correlation, background-subtracted fluxes.",
                color: "#ff6b6b",
                note: "17 features total",
              },
              {
                title: "Transfer Learning Ready",
                desc: "Pre-train on GOES XRS data, fine-tune on Aditya-L1. Not yet implemented.",
                color: "#00ffaa",
                note: "Requires historical GOES data",
              },
            ].map((feat, i) => (
              <AnimatedCard key={i} delay={i * 0.05}>
                <motion.div
                  whileHover={{ y: -4, borderColor: `${feat.color}30` }}
                  className="glass-card p-6 h-full group cursor-default relative overflow-hidden"
                >
                  <div
                    className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-700"
                    style={{ background: `radial-gradient(circle at 30% 30%, ${feat.color}06, transparent 70%)` }}
                  />
                  <div className="relative">
                    <div className="w-2 h-2 rounded-full mb-4" style={{ background: feat.color }} />
                    <h3 className="text-lg font-semibold text-white mb-2">{feat.title}</h3>
                    <p className="text-sm text-white/40 leading-relaxed mb-3">{feat.desc}</p>
                    <div className="text-[10px] text-white/25 font-mono uppercase tracking-wider">{feat.note}</div>
                  </div>
                </motion.div>
              </AnimatedCard>
            ))}
          </div>
        </div>
      </section>

      <GlowLine />

      {/* Data Status */}
      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto">
          <AnimatedCard>
            <div className="text-left mb-12">
              <div className="text-xs text-white/40 uppercase tracking-[0.2em] mb-3">Data</div>
              <h2 className="text-3xl md:text-5xl font-bold tracking-tight">
                Current <span className="gradient-text">Status</span>
              </h2>
            </div>
          </AnimatedCard>

          <div className="grid md:grid-cols-2 gap-6">
            <AnimatedCard delay={0.1}>
              <div className="glass-card p-6">
                <div className="text-xs text-white/40 uppercase tracking-wider mb-3">SoLEXS</div>
                <div className="text-lg font-semibold text-white mb-1">June 14, 2026</div>
                <div className="text-sm text-white/40">86,400 points (1s cadence, 24 hours)</div>
                <div className="text-xs text-white/25 mt-2">Energy range: 2-22 keV</div>
              </div>
            </AnimatedCard>
            <AnimatedCard delay={0.15}>
              <div className="glass-card p-6">
                <div className="text-xs text-white/40 uppercase tracking-wider mb-3">HEL1OS</div>
                <div className="text-lg font-semibold text-white mb-1">June 15, 2026</div>
                <div className="text-sm text-white/40">43,182 points (1s cadence, 12 hours)</div>
                <div className="text-xs text-white/25 mt-2">Energy range: 8-150 keV</div>
              </div>
            </AnimatedCard>
          </div>

          <AnimatedCard delay={0.2}>
            <div className="mt-6 glass-card p-4 border-[#ffa502]/20">
              <div className="text-xs text-[#ffa502]">
                ⚠ No temporal overlap between instruments. Download more data from
                <a href="https://pradan.issdc.gov.in/al1" target="_blank" rel="noopener noreferrer"
                   className="underline ml-1 text-[#ffa502]/80 hover:text-[#ffa502]">
                  PRADAN portal
                </a>
                for full dual-instrument operation.
              </div>
            </div>
          </AnimatedCard>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 border-t border-white/5">
        <div className="max-w-6xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="text-xs text-white/30 uppercase tracking-[0.2em]">
            Bharatiya Antariksh Hackathon 2026 &middot; Problem Statement 15
          </div>
          <div className="text-xs text-white/20">
            Aditya-L1 SoLEXS + HEL1OS &middot; Parallel CNN-BiLSTM &middot; PRADAN Data
          </div>
        </div>
      </footer>
    </div>
  )
}
