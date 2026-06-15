import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Activity, Eye, EyeOff, Mail, Lock, ArrowRight, ShieldCheck } from "lucide-react";
import { toast } from "sonner";

const MOCK_ADMINS = [
  { email: "admin@kneexpert.com", password: "admin123", name: "System Admin" },
  { email: "superadmin@kneexpert.com", password: "admin123", name: "Super Admin" },
];

export default function AdminLoginPage() {
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
      const admin = MOCK_ADMINS.find(
        (a) => a.email === email && a.password === password
      );
      setIsLoading(false);
      if (admin) {
        toast.success(`Welcome, ${admin.name}!`);
        navigate("/admin");
      } else {
        setError("Invalid admin credentials. Try one of the demo accounts below.");
      }
    }, 1000);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-sidebar p-6">
      <motion.div
        initial={{ opacity: 0, y: 16 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.4 }}
        className="w-full max-w-sm"
      >
        <div className="flex flex-col items-center mb-8">
          <div className="w-12 h-12 rounded-xl bg-primary flex items-center justify-center mb-3">
            <Activity className="w-6 h-6 text-primary-foreground" />
          </div>
          <span className="text-lg font-semibold text-sidebar-accent-foreground">KneeXpert</span>
          <div className="flex items-center gap-1.5 mt-1.5">
            <ShieldCheck className="w-3.5 h-3.5 text-primary" />
            <span className="text-xs text-sidebar-foreground/60 font-medium">Admin Portal</span>
          </div>
        </div>

        <div className="bg-background rounded-2xl p-6 shadow-lg">
          <h1 className="text-lg font-semibold tracking-tight mb-1">Admin Sign In</h1>
          <p className="text-xs text-muted-foreground mb-5">Access the system administration panel</p>

          <form onSubmit={handleLogin} className="space-y-4">
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Email</label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <input
                  type="email"
                  value={email}
                  onChange={e => { setEmail(e.target.value); setError(""); }}
                  placeholder="admin@kneexpert.com"
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

            <button
              type="submit"
              disabled={isLoading}
              className="w-full flex items-center justify-center gap-2 py-2.5 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors disabled:opacity-50"
            >
              {isLoading ? (
                <div className="w-4 h-4 border-2 border-primary-foreground/30 border-t-primary-foreground rounded-full animate-spin" />
              ) : (
                <>
                  Sign In as Admin
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </form>

          {/* Demo accounts */}
          <div className="mt-4 p-3 rounded-lg border border-dashed bg-muted/30">
            <p className="text-[10px] font-semibold text-muted-foreground uppercase tracking-wider mb-2">Demo Accounts</p>
            <div className="space-y-1.5">
              {MOCK_ADMINS.map((a) => (
                <button
                  key={a.email}
                  type="button"
                  onClick={() => { setEmail(a.email); setPassword(a.password); setError(""); }}
                  className="w-full flex items-center justify-between text-left px-2.5 py-1.5 rounded-md hover:bg-muted transition-colors group"
                >
                  <span className="text-xs text-foreground">{a.name}</span>
                  <span className="text-[10px] text-muted-foreground group-hover:text-foreground transition-colors">{a.email}</span>
                </button>
              ))}
            </div>
          </div>
        </div>

        <p className="text-xs text-sidebar-foreground/40 text-center mt-5">
          <Link to="/login" className="text-sidebar-foreground/60 hover:text-sidebar-accent-foreground transition-colors">
            ← Back to Doctor Login
          </Link>
        </p>
      </motion.div>
    </div>
  );
}
