import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion } from "framer-motion";
import { Activity, Eye, EyeOff, Mail, Lock, User, Building, ArrowRight } from "lucide-react";

export default function SignupPage() {
  const navigate = useNavigate();
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [institution, setInstitution] = useState("");
  const [specialty, setSpecialty] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);

  const handleSignup = (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setTimeout(() => {
      setIsLoading(false);
      navigate("/");
    }, 1200);
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
            Join the Clinical<br />AI Workspace
          </h2>
          <p className="text-sm text-sidebar-foreground/70 leading-relaxed max-w-sm">
            Request access to KneeXpert's AI-powered diagnostic platform. Your account will be reviewed and approved by an administrator.
          </p>
          <div className="mt-6 space-y-3">
            {["Ensemble AI with 95.1% accuracy", "Grad-CAM transparency for every diagnosis", "HIPAA-compliant data handling"].map(feat => (
              <div key={feat} className="flex items-center gap-2.5">
                <div className="w-1.5 h-1.5 rounded-full bg-primary" />
                <span className="text-xs text-sidebar-foreground/60">{feat}</span>
              </div>
            ))}
          </div>
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
          <div className="lg:hidden flex items-center gap-3 mb-8">
            <div className="w-9 h-9 rounded-lg bg-primary flex items-center justify-center">
              <Activity className="w-4 h-4 text-primary-foreground" />
            </div>
            <span className="text-base font-semibold">KneeXpert</span>
          </div>

          <h1 className="text-xl font-semibold tracking-tight mb-1">Request Access</h1>
          <p className="text-sm text-muted-foreground mb-6">Create your clinical account</p>

          <form onSubmit={handleSignup} className="space-y-3.5">
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Full Name</label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <input
                  type="text"
                  value={fullName}
                  onChange={e => setFullName(e.target.value)}
                  placeholder="Dr. Nguyễn Văn A"
                  className="w-full pl-10 pr-4 py-2.5 rounded-lg border bg-background text-sm placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-ring/30 transition"
                  required
                />
              </div>
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Email</label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <input
                  type="email"
                  value={email}
                  onChange={e => setEmail(e.target.value)}
                  placeholder="doctor@hospital.edu.vn"
                  className="w-full pl-10 pr-4 py-2.5 rounded-lg border bg-background text-sm placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-ring/30 transition"
                  required
                />
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Specialty</label>
                <select
                  value={specialty}
                  onChange={e => setSpecialty(e.target.value)}
                  className="w-full px-3 py-2.5 rounded-lg border bg-background text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring/30 transition"
                  required
                >
                  <option value="">Select...</option>
                  <option value="radiology">Radiology</option>
                  <option value="orthopedics">Orthopedics</option>
                  <option value="rheumatology">Rheumatology</option>
                  <option value="sports-medicine">Sports Medicine</option>
                  <option value="other">Other</option>
                </select>
              </div>
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Institution</label>
                <div className="relative">
                  <Building className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                  <input
                    type="text"
                    value={institution}
                    onChange={e => setInstitution(e.target.value)}
                    placeholder="Hospital"
                    className="w-full pl-10 pr-3 py-2.5 rounded-lg border bg-background text-sm placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-ring/30 transition"
                    required
                  />
                </div>
              </div>
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Password</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <input
                  type={showPassword ? "text" : "password"}
                  value={password}
                  onChange={e => setPassword(e.target.value)}
                  placeholder="Min. 8 characters"
                  className="w-full pl-10 pr-10 py-2.5 rounded-lg border bg-background text-sm placeholder:text-muted-foreground/50 focus:outline-none focus:ring-2 focus:ring-ring/30 transition"
                  required
                  minLength={8}
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

            <div className="flex items-start gap-2 pt-1">
              <input type="checkbox" className="w-3.5 h-3.5 rounded border-border accent-primary mt-0.5" required />
              <span className="text-[11px] text-muted-foreground leading-tight">
                I agree to the Terms of Service and acknowledge that patient data will be handled in compliance with HIPAA regulations.
              </span>
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
                  Create Account
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </form>

          <p className="text-xs text-muted-foreground text-center mt-5">
            Already have an account?{" "}
            <Link to="/login" className="text-primary hover:text-primary/80 font-medium transition-colors">
              Sign in
            </Link>
          </p>
        </motion.div>
      </div>
    </div>
  );
}
