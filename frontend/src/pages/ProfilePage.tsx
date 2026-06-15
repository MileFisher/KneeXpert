import { useState } from "react";
import { motion } from "framer-motion";
import { User, Mail, Building2, Award, Shield, Camera } from "lucide-react";

const containerVariants = {
  hidden: { opacity: 0, y: 8 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.3, staggerChildren: 0.05 } },
};
const itemVariants = { hidden: { opacity: 0, y: 4 }, visible: { opacity: 1, y: 0 } };

export default function ProfilePage() {
  const [doctorName, setDoctorName] = useState("Dr. Quốc Châu");
  const [specialty, setSpecialty] = useState("Orthopedic Radiology");
  const [institution, setInstitution] = useState("Hanoi Medical University Hospital");
  const [email, setEmail] = useState("chau.nguyen@hmu.edu.vn");
  const [phone, setPhone] = useState("+84 912 345 678");
  const [bio, setBio] = useState("Specialized in musculoskeletal radiology with focus on AI-assisted knee osteoarthritis diagnosis. 12+ years of clinical experience.");

  return (
    <div className="min-h-screen overflow-auto">
      <motion.div variants={containerVariants} initial="hidden" animate="visible" className="p-4 sm:p-6 max-w-3xl mx-auto">
        <motion.div variants={itemVariants} className="mb-6">
          <h1 className="text-xl sm:text-2xl font-semibold tracking-tight">My Profile</h1>
          <p className="text-xs text-muted-foreground mt-0.5">Manage your professional information</p>
        </motion.div>

        {/* Avatar card */}
        <motion.div variants={itemVariants} className="card-clinical mb-6">
          <div className="flex items-center gap-5">
            <div className="relative group">
              <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-primary/80 to-primary flex items-center justify-center text-primary-foreground text-2xl font-semibold">
                QC
              </div>
              <button className="absolute inset-0 rounded-2xl bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                <Camera className="w-5 h-5 text-white" />
              </button>
            </div>
            <div>
              <h2 className="text-lg font-semibold">{doctorName}</h2>
              <p className="text-sm text-muted-foreground">{specialty}</p>
              <div className="flex items-center gap-2 mt-1.5">
                <span className="inline-flex items-center gap-1 text-[10px] font-medium px-2 py-0.5 rounded-full bg-success/10 text-success">
                  <Shield className="w-3 h-3" /> Verified
                </span>
                <span className="text-[10px] text-muted-foreground">License: VN-RAD-2015-08842</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Info form */}
        <motion.div variants={itemVariants} className="card-clinical mb-6">
          <p className="text-sm font-medium mb-4">Personal Information</p>
          <div className="space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-1.5 block flex items-center gap-1.5"><User className="w-3 h-3" />Full Name</label>
                <input value={doctorName} onChange={e => setDoctorName(e.target.value)} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20" />
              </div>
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-1.5 block flex items-center gap-1.5"><Award className="w-3 h-3" />Specialty</label>
                <input value={specialty} onChange={e => setSpecialty(e.target.value)} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20" />
              </div>
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1.5 block flex items-center gap-1.5"><Building2 className="w-3 h-3" />Institution</label>
              <input value={institution} onChange={e => setInstitution(e.target.value)} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20" />
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-1.5 block flex items-center gap-1.5"><Mail className="w-3 h-3" />Email</label>
                <input type="email" value={email} onChange={e => setEmail(e.target.value)} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20" />
              </div>
              <div>
                <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Phone</label>
                <input value={phone} onChange={e => setPhone(e.target.value)} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20" />
              </div>
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Bio</label>
              <textarea value={bio} onChange={e => setBio(e.target.value)} rows={3} className="w-full px-3 py-2 rounded-lg border bg-background text-sm resize-none focus:outline-none focus:ring-2 focus:ring-ring/20" />
            </div>
            <div>
              <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Medical License Number</label>
              <input defaultValue="VN-RAD-2015-08842" className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20" />
            </div>
          </div>
          <button className="mt-4 px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors">
            Save Changes
          </button>
        </motion.div>

        {/* Stats */}
        <motion.div variants={itemVariants} className="card-clinical">
          <p className="text-sm font-medium mb-4">Activity Summary</p>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[
              { label: "Scans Reviewed", value: "1,247" },
              { label: "Reports Generated", value: "892" },
              { label: "Cases Confirmed", value: "1,105" },
              { label: "Avg. Response Time", value: "4.2 min" },
            ].map(s => (
              <div key={s.label} className="p-3 rounded-lg bg-muted/50 text-center">
                <p className="text-lg font-semibold">{s.value}</p>
                <p className="text-[10px] text-muted-foreground mt-0.5">{s.label}</p>
              </div>
            ))}
          </div>
        </motion.div>
      </motion.div>
    </div>
  );
}
