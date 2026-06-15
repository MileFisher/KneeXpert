import { useState, useEffect } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, ChevronRight, ChevronLeft, Sparkles, LayoutDashboard, Users, Scan, FileText, Settings, CheckCircle2 } from "lucide-react";
import { cn } from "@/lib/utils";

const tutorialSteps = [
  {
    title: "Welcome to KneeXpert! 👋",
    description: "Let's walk you through the key features of this AI-assisted knee diagnostics platform. This tour takes about 1 minute.",
    icon: Sparkles,
    color: "text-primary",
    bg: "bg-primary/10",
  },
  {
    title: "Dashboard Overview",
    description: "Your dashboard shows daily stats, recent activity, and patient summaries at a glance. Monitor pending scans, flagged cases, and diagnostic trends here.",
    icon: LayoutDashboard,
    color: "text-primary",
    bg: "bg-primary/10",
    path: "/",
  },
  {
    title: "Patient Management",
    description: "View all patients, filter by status or KL grade, and access detailed patient profiles. Click any patient to see their full history and scans.",
    icon: Users,
    color: "text-success",
    bg: "bg-success/10",
    path: "/patients",
  },
  {
    title: "Diagnostic Workspace",
    description: "Select a patient, upload X-ray or MRI scans, and let the AI analyze them. Use tools like zoom, pan, measure, and annotate. Compare original scans with Grad-CAM heatmaps side by side.",
    icon: Scan,
    color: "text-warning",
    bg: "bg-warning/10",
    path: "/diagnostics",
  },
  {
    title: "Reports Hub",
    description: "Access all diagnostic reports, track priority reviews, and export PDF reports. Monitor patient progression and clinical trends over time.",
    icon: FileText,
    color: "text-destructive",
    bg: "bg-destructive/10",
    path: "/reports",
  },
  {
    title: "Settings & Preferences",
    description: "Customize your profile, notification preferences, display settings, and security options. The app adapts to your workflow.",
    icon: Settings,
    color: "text-muted-foreground",
    bg: "bg-muted",
    path: "/settings",
  },
  {
    title: "You're All Set! 🎉",
    description: "You can revisit this tutorial anytime from Settings > Help. Start by exploring the Dashboard or selecting a patient for diagnosis.",
    icon: CheckCircle2,
    color: "text-success",
    bg: "bg-success/10",
  },
];

export function TutorialOverlay() {
  const [visible, setVisible] = useState(false);
  const [step, setStep] = useState(0);

  useEffect(() => {
    const dismissed = localStorage.getItem("kneexpert-tutorial-dismissed");
    if (!dismissed) {
      setVisible(true);
    }
  }, []);

  const dismiss = () => {
    setVisible(false);
    localStorage.setItem("kneexpert-tutorial-dismissed", "true");
  };

  const next = () => {
    if (step < tutorialSteps.length - 1) setStep(step + 1);
    else dismiss();
  };

  const prev = () => {
    if (step > 0) setStep(step - 1);
  };

  const current = tutorialSteps[step];

  if (!visible) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-[100] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4"
      >
        <motion.div
          key={step}
          initial={{ opacity: 0, scale: 0.95, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.95, y: -10 }}
          transition={{ duration: 0.25 }}
          className="w-full max-w-md bg-card rounded-2xl shadow-2xl border overflow-hidden"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-5 pt-4">
            <div className="flex items-center gap-1.5">
              {tutorialSteps.map((_, i) => (
                <div
                  key={i}
                  className={cn(
                    "h-1 rounded-full transition-all duration-300",
                    i === step ? "w-6 bg-primary" : i < step ? "w-3 bg-primary/40" : "w-3 bg-muted"
                  )}
                />
              ))}
            </div>
            <button
              onClick={dismiss}
              className="p-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Content */}
          <div className="px-5 py-6">
            <div className={cn("w-14 h-14 rounded-xl flex items-center justify-center mb-4", current.bg)}>
              <current.icon className={cn("w-7 h-7", current.color)} />
            </div>
            <h2 className="text-lg font-semibold mb-2">{current.title}</h2>
            <p className="text-sm text-muted-foreground leading-relaxed">{current.description}</p>
          </div>

          {/* Footer */}
          <div className="px-5 pb-5 flex items-center justify-between">
            <button
              onClick={dismiss}
              className="text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              Skip tutorial
            </button>
            <div className="flex items-center gap-2">
              {step > 0 && (
                <button
                  onClick={prev}
                  className="inline-flex items-center gap-1 px-3 py-2 rounded-lg border text-sm font-medium hover:bg-muted transition-colors"
                >
                  <ChevronLeft className="w-3.5 h-3.5" />
                  Back
                </button>
              )}
              <button
                onClick={next}
                className="inline-flex items-center gap-1 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors"
              >
                {step === tutorialSteps.length - 1 ? "Get Started" : "Next"}
                {step < tutorialSteps.length - 1 && <ChevronRight className="w-3.5 h-3.5" />}
              </button>
            </div>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
}

export function resetTutorial() {
  localStorage.removeItem("kneexpert-tutorial-dismissed");
}
