import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Activity, Eye, EyeOff, Mail, Lock, ArrowRight } from "lucide-react";
import { toast } from "sonner";

const MOCK_DOCTORS = [
  { email: "chau@kneexpert.com", password: "doctor123", name: "Dr. Quốc Châu" },
  { email: "linh@kneexpert.com", password: "doctor123", name: "Dr. Thu Linh" },
  { email: "minh@kneexpert.com", password: "doctor123", name: "Dr. Anh Minh" },
];

export default function LoginPage() {
  const navigate = useNavigate();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    setTimeout(() => {
      const doctor = MOCK_DOCTORS.find(
        (d) => d.email === email && d.password === password
      );
      setIsLoading(false);
      if (doctor) {
        toast.success(`Welcome back, ${doctor.name}!`);
        navigate("/");
      } else {
        setError("Invalid email or password. Try one of the demo accounts below.");
      }
    }, 1000);
  };

  return (
    <div className="min-h-screen flex bg-background">
      {/* Left - Branding */}
      <div className="hidden lg:flex lg:w-[480px] xl:w-[560px] bg-sidebar flex-col justify-between p-10">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-primary flex items-center justify-center">
            <Activity className="w-5 h-5 text-primary-foreground" />
          </div>
          <span className="text-lg font-semibold text-sidebar-accent-foreground">KneeXpert</span>
        </div>
        <div>
          <h2 className="text-2xl font-semibold text-sidebar-accent-foreground leading-tight mb-3">
            AI-Powered Knee<br />Osteoarthritis Diagnostics
          </h2>
          <p className="text-sm text-sidebar-foreground/70 leading-relaxed max-w-sm">
            Ensemble deep learning with Grad-CAM visualization for transparent, physician-trusted KL grading across X-ray and MRI modalities.
          </p>
        </div>
        <p className="text-[11px] text-sidebar-foreground/40">© 2026 KneeXpert · Hanoi Medical University</p>
      </div>

      {/* Right - Form */}
      <div className="flex-1 flex items-center justify-center p-6">
        <motion.div
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
          className="w-full max-w-sm"
        >
          {/* Mobile logo */}
          <div className="lg:hidden flex items-center gap-3 mb-8">
            <div className="w-9 h-9 rounded-lg bg-primary flex items-center justify-center">
              <Activity className="w-4 h-4 text-primary-foreground" />
            </div>
            <span className="text-base font-semibold">KneeXpert</span>
          </div>

          <h1 className="text-xl font-semibold tracking-tight mb-1">Welcome back</h1>
          <p className="text-sm text-muted-foreground mb-6">Sign in to your clinical workspace</p>

          <form onSubmit={handleLogin} className="space-y-4">
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Email</label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <input
                  type="email"
                  value={email}
                  onChange={e => { setEmail(e.target.value); setError(""); }}
                  placeholder="doctor@hospital.edu.vn"
                  className="w-full pl-10 pr-4 py-2.5 rounded-lg border bg-background text-sm placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-ring/30 transition"
                  required
                />
              </div>
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Password</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={e => { setPassword(e.target.value); setError(""); }}
                  placeholder="••••••••"
                  className="w-full pl-10 pr-10 py-2.5 rounded-lg border bg-background text-sm placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-ring/30 transition"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
                >
                  {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                </button>
              </div>
            </div>

            {error && (
              <p className="text-xs text-destructive bg-destructive/10 rounded-lg px-3 py-2">{error}</p>
            )}

            <div className="flex items-center justify-between">
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" className="w-3.5 h-3.5 rounded border-border accent-primary" />
                <span className="text-xs text-muted-foreground">Remember me</span>
              </label>
              <button type="button" className="text-xs text-primary hover:text-primary/80 transition-colors">
                Forgot password?
              </button>
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-50"
            >
              {isLoading ? (
                <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
              ) : (
                <>
                  Sign In
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </form>

          {/* Demo accounts */}
          <div className="mt-5 p-3 rounded-lg border border-dashed bg-muted/30">
            <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">Demo Accounts</p>
            <div className="space-y-1.5">
              {MOCK_DOCTORS.map((d) => (
                <button
                  key={d.email}
                  type="button"
                  onClick={() => { setEmail(d.email); setPassword(d.password); setError(""); }}
                  className="w-full flex items-center justify-between text-left px-2.5 py-1.5 rounded-md hover:bg-muted transition-colors group"
                >
                  <span className="text-xs text-foreground">{d.name}</span>
                  <span className="text-[10px] text-muted-foreground group-hover:text-foreground transition-colors">{d.email}</span>
                </button>
              ))}
            </div>
          </div>

          <p className="text-xs text-muted-foreground text-center mt-6">
            Don't have an account?{" "}
            <Link to="/signup" className="text-primary hover:text-primary/80 font-medium transition-colors">
              Request access
            </Link>
          </p>

          <div className="mt-6 pt-6 border-t">
            <Link
              to="/admin/login"
              className="flex items-center justify-center gap-2 w-full py-2 rounded-lg border text-xs text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
            >
              Admin Portal
              <ArrowRight className="w-3 h-3" />
            </Link>
          </div>
        </motion.div>
      </div>
    </div>
  );
}
