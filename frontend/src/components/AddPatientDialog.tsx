import { useState } from "react";
import { X, Upload, Plus, User, Activity, FileText, Stethoscope } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import type { NewPatientInput } from "@/context/PatientContext";

interface AddPatientDialogProps {
  open: boolean;
  onClose: () => void;
  onAddPatient: (input: NewPatientInput) => string;
}

const steps = [
  { id: 1, label: "Demographics", icon: User },
  { id: 2, label: "Medical History", icon: FileText },
  { id: 3, label: "Symptoms & Exam", icon: Stethoscope },
  { id: 4, label: "Imaging Upload", icon: Upload },
];

export function AddPatientDialog({ open, onClose, onAddPatient }: AddPatientDialogProps) {
  const [step, setStep] = useState(1);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState({
    firstName: "", lastName: "", age: "", gender: "Male",
    height: "", weight: "",
    phone: "", email: "",
    medicalHistory: "", surgicalHistory: "", familyHistory: "",
    medications: "", allergies: "",
    chiefComplaint: "", symptomDuration: "", painLevel: 5,
    morningStiffness: false, crepitus: false, swelling: false, reducedROM: false,
    affectedKnee: "right",
    imagingType: "xray",
    notes: "",
  });

  if (!open) return null;

  const updateForm = (field: string, value: any) => {
    setError(null);
    setForm(prev => ({ ...prev, [field]: value }));
  };

  const handleCreate = () => {
    // Validate required fields
    if (!form.firstName.trim() || !form.lastName.trim()) {
      setError("First and last name are required.");
      setStep(1);
      return;
    }
    if (!form.age || Number(form.age) < 1 || Number(form.age) > 120) {
      setError("Please enter a valid age.");
      setStep(1);
      return;
    }
    if (!form.chiefComplaint.trim()) {
      setError("Chief complaint is required.");
      setStep(3);
      return;
    }

    const heightM = Number(form.height) / 100;
    const bmi = heightM > 0 && Number(form.weight) > 0
      ? Math.round((Number(form.weight) / (heightM * heightM)) * 10) / 10
      : 25.0;

    // Build history and symptoms from form
    const historyParts: string[] = [];
    if (form.medicalHistory.trim()) historyParts.push(form.medicalHistory.trim());
    if (form.surgicalHistory.trim()) historyParts.push(`Surgical: ${form.surgicalHistory.trim()}`);
    if (form.familyHistory.trim()) historyParts.push(`Family Hx: ${form.familyHistory.trim()}`);
    if (form.medications.trim()) historyParts.push(`Medications: ${form.medications.trim()}`);
    if (form.allergies.trim()) historyParts.push(`Allergies: ${form.allergies.trim()}`);

    const symptomParts: string[] = [];
    if (form.chiefComplaint.trim()) symptomParts.push(form.chiefComplaint.trim());
    if (form.symptomDuration.trim()) symptomParts.push(`Duration: ${form.symptomDuration.trim()}`);
    const examFindings: string[] = [];
    if (form.morningStiffness) examFindings.push("Morning stiffness >30min");
    if (form.crepitus) examFindings.push("Crepitus on flexion");
    if (form.swelling) examFindings.push("Joint swelling");
    if (form.reducedROM) examFindings.push("Reduced ROM");
    if (examFindings.length) symptomParts.push(`Exam: ${examFindings.join(", ")}`);

    const kneeLabel = form.affectedKnee === "bilateral" ? "Bilateral Knee" :
      form.affectedKnee === "left" ? "Left Knee" : "Right Knee";

    const viewMap: Record<string, string> = { xray: "AP", mri: "Sagittal", ct: "Axial" };

    onAddPatient({
      name: `${form.firstName.trim()} ${form.lastName.trim()}`,
      age: Number(form.age),
      gender: form.gender as "Male" | "Female",
      bmi,
      history: historyParts.join(". ") || "No history recorded.",
      symptoms: symptomParts.join(". ") || "No symptoms recorded.",
      painLevel: form.painLevel,
      modality: form.imagingType as "xray" | "mri",
      region: kneeLabel,
      view: viewMap[form.imagingType] ?? "AP",
    });

    // Reset and close
    setForm({
      firstName: "", lastName: "", age: "", gender: "Male",
      height: "", weight: "", phone: "", email: "",
      medicalHistory: "", surgicalHistory: "", familyHistory: "",
      medications: "", allergies: "",
      chiefComplaint: "", symptomDuration: "", painLevel: 5,
      morningStiffness: false, crepitus: false, swelling: false, reducedROM: false,
      affectedKnee: "right", imagingType: "xray", notes: "",
    });
    setStep(1);
    setError(null);
    onClose();
  };

  return (
    <AnimatePresence>
      <div className="fixed inset-0 z-50 flex items-center justify-center">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="absolute inset-0 bg-foreground/40 backdrop-blur-sm"
          onClick={onClose}
        />
        <motion.div
          initial={{ opacity: 0, scale: 0.96, y: 10 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.96, y: 10 }}
          transition={{ duration: 0.2, ease: [0.25, 0.1, 0.25, 1] }}
          className="relative w-[720px] max-h-[85vh] bg-background rounded-2xl shadow-xl border overflow-hidden flex flex-col"
        >
          {/* Header */}
          <div className="flex items-center justify-between px-6 py-4 border-b">
            <div>
              <h2 className="text-lg font-medium">New Patient Intake</h2>
              <p className="text-xs text-muted-foreground mt-0.5">Complete all sections for comprehensive EHR</p>
            </div>
            <button onClick={onClose} className="p-1.5 rounded-lg hover:bg-muted transition-colors">
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Step Indicator */}
          <div className="flex items-center gap-1 px-6 py-3 border-b bg-muted/30">
            {steps.map((s, i) => (
              <button
                key={s.id}
                onClick={() => setStep(s.id)}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                  step === s.id
                    ? "bg-primary text-primary-foreground"
                    : step > s.id
                    ? "bg-primary-muted text-primary"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <s.icon className="w-3.5 h-3.5" />
                {s.label}
              </button>
            ))}
          </div>

          {/* Form Content */}
          <div className="flex-1 overflow-auto px-6 py-5">
            <AnimatePresence mode="wait">
              <motion.div
                key={step}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20 }}
                transition={{ duration: 0.15 }}
              >
                {step === 1 && (
                  <div className="space-y-4">
                    <p className="section-header mb-4">Patient Demographics</p>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="text-xs font-medium text-muted-foreground mb-1.5 block">First Name *</label>
                        <input value={form.firstName} onChange={e => updateForm("firstName", e.target.value)} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20" placeholder="Nguyễn" />
                      </div>
                      <div>
                        <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Last Name *</label>
                        <input value={form.lastName} onChange={e => updateForm("lastName", e.target.value)} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20" placeholder="Văn An" />
                      </div>
                    </div>
                    <div className="grid grid-cols-3 gap-4">
                      <div>
                        <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Age *</label>
                        <input type="number" value={form.age} onChange={e => updateForm("age", e.target.value)} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20" placeholder="65" />
                      </div>
                      <div>
                        <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Gender *</label>
                        <select value={form.gender} onChange={e => updateForm("gender", e.target.value)} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20">
                          <option>Male</option>
                          <option>Female</option>
                        </select>
                      </div>
                      <div>
                        <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Affected Knee *</label>
                        <select value={form.affectedKnee} onChange={e => updateForm("affectedKnee", e.target.value)} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20">
                          <option value="right">Right</option>
                          <option value="left">Left</option>
                          <option value="bilateral">Bilateral</option>
                        </select>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Height (cm)</label>
                        <input type="number" value={form.height} onChange={e => updateForm("height", e.target.value)} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20" placeholder="170" />
                      </div>
                      <div>
                        <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Weight (kg)</label>
                        <input type="number" value={form.weight} onChange={e => updateForm("weight", e.target.value)} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20" placeholder="78" />
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Phone</label>
                        <input value={form.phone} onChange={e => updateForm("phone", e.target.value)} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20" placeholder="+84 xxx xxx xxx" />
                      </div>
                      <div>
                        <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Email</label>
                        <input type="email" value={form.email} onChange={e => updateForm("email", e.target.value)} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20" placeholder="patient@email.com" />
                      </div>
                    </div>
                  </div>
                )}

                {step === 2 && (
                  <div className="space-y-4">
                    <p className="section-header mb-4">Medical & Surgical History</p>
                    <div>
                      <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Medical History *</label>
                      <textarea value={form.medicalHistory} onChange={e => updateForm("medicalHistory", e.target.value)} rows={3} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20 resize-none" placeholder="Prior knee conditions, chronic diseases, diabetes, hypertension..." />
                    </div>
                    <div>
                      <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Surgical History</label>
                      <textarea value={form.surgicalHistory} onChange={e => updateForm("surgicalHistory", e.target.value)} rows={2} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20 resize-none" placeholder="Previous knee surgeries, meniscus repair, ACL reconstruction..." />
                    </div>
                    <div>
                      <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Family History</label>
                      <textarea value={form.familyHistory} onChange={e => updateForm("familyHistory", e.target.value)} rows={2} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20 resize-none" placeholder="Family history of osteoarthritis, rheumatoid arthritis..." />
                    </div>
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Current Medications</label>
                        <textarea value={form.medications} onChange={e => updateForm("medications", e.target.value)} rows={2} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20 resize-none" placeholder="NSAIDs, corticosteroids..." />
                      </div>
                      <div>
                        <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Allergies</label>
                        <textarea value={form.allergies} onChange={e => updateForm("allergies", e.target.value)} rows={2} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20 resize-none" placeholder="Drug allergies, contrast dye reactions..." />
                      </div>
                    </div>
                  </div>
                )}

                {step === 3 && (
                  <div className="space-y-4">
                    <p className="section-header mb-4">Symptoms & Physical Examination</p>
                    <div>
                      <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Chief Complaint *</label>
                      <textarea value={form.chiefComplaint} onChange={e => updateForm("chiefComplaint", e.target.value)} rows={2} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20 resize-none" placeholder="Describe the primary complaint..." />
                    </div>
                    <div>
                      <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Symptom Duration</label>
                      <input value={form.symptomDuration} onChange={e => updateForm("symptomDuration", e.target.value)} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20" placeholder="e.g., 6 months, 2 years" />
                    </div>

                    <div>
                      <label className="text-xs font-medium text-muted-foreground mb-2 block">Pain Level: {form.painLevel}/10</label>
                      <div className="flex items-center gap-3">
                        <span className="text-xs text-muted-foreground">0</span>
                        <input
                          type="range" min="0" max="10" value={form.painLevel}
                          onChange={e => updateForm("painLevel", parseInt(e.target.value))}
                          className="flex-1 accent-primary h-2"
                        />
                        <span className="text-xs text-muted-foreground">10</span>
                      </div>
                      <div className="flex justify-between mt-1">
                        <span className="text-[10px] text-muted-foreground">No pain</span>
                        <span className="text-[10px] text-muted-foreground">Severe</span>
                      </div>
                    </div>

                    <div>
                      <label className="text-xs font-medium text-muted-foreground mb-2 block">Physical Examination Findings</label>
                      <div className="grid grid-cols-2 gap-2">
                        {[
                          { key: "morningStiffness", label: "Morning Stiffness (>30min)" },
                          { key: "crepitus", label: "Crepitus on Flexion" },
                          { key: "swelling", label: "Joint Swelling" },
                          { key: "reducedROM", label: "Reduced Range of Motion" },
                        ].map(item => (
                          <label key={item.key} className="flex items-center gap-2 px-3 py-2 rounded-lg border cursor-pointer hover:bg-muted/50 transition-colors">
                            <input
                              type="checkbox"
                              checked={(form as any)[item.key]}
                              onChange={e => updateForm(item.key, e.target.checked)}
                              className="rounded accent-primary"
                            />
                            <span className="text-xs">{item.label}</span>
                          </label>
                        ))}
                      </div>
                    </div>

                    <div>
                      <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Additional Notes</label>
                      <textarea value={form.notes} onChange={e => updateForm("notes", e.target.value)} rows={2} className="w-full px-3 py-2 rounded-lg border bg-background text-sm focus:outline-none focus:ring-2 focus:ring-ring/20 resize-none" placeholder="Other clinical observations..." />
                    </div>
                  </div>
                )}

                {step === 4 && (
                  <div className="space-y-4">
                    <p className="section-header mb-4">Imaging Upload</p>

                    <div>
                      <label className="text-xs font-medium text-muted-foreground mb-2 block">Imaging Modality</label>
                      <div className="flex gap-2">
                        {[
                          { id: "xray", label: "X-ray" },
                          { id: "mri", label: "MRI" },
                          { id: "ct", label: "CT Scan" },
                        ].map(type => (
                          <button
                            key={type.id}
                            onClick={() => updateForm("imagingType", type.id)}
                            className={`px-4 py-2 rounded-lg text-sm font-medium border transition-all ${
                              form.imagingType === type.id
                                ? "bg-primary text-primary-foreground border-primary"
                                : "hover:bg-muted"
                            }`}
                          >
                            {type.label}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Upload Area */}
                    <div className="border-2 border-dashed rounded-xl p-8 text-center hover:border-primary/40 hover:bg-primary-muted/30 transition-all cursor-pointer group">
                      <div className="w-14 h-14 rounded-xl bg-muted group-hover:bg-primary-muted mx-auto mb-3 flex items-center justify-center transition-colors">
                        <Upload className="w-6 h-6 text-muted-foreground group-hover:text-primary transition-colors" />
                      </div>
                      <p className="text-sm font-medium">Drop {form.imagingType === "xray" ? "X-ray" : form.imagingType === "mri" ? "MRI" : "CT"} images here</p>
                      <p className="text-xs text-muted-foreground mt-1">Supports DICOM, PNG, JPEG · Max 50MB per file</p>
                      <button className="mt-3 px-4 py-1.5 rounded-lg bg-primary text-primary-foreground text-xs font-medium hover:bg-primary/90 transition-colors">
                        Browse Files
                      </button>
                    </div>

                    <div className="bg-primary-muted/50 rounded-lg p-3">
                      <p className="text-xs font-medium text-primary mb-1">Pre-processing Pipeline</p>
                      <p className="text-xs text-muted-foreground leading-relaxed">
                        Uploaded images will be automatically processed through our pipeline: DICOM conversion → Noise reduction → Normalization → AI-ready format. CLAHE enhancement and GAN-based artifact removal are applied for optimal diagnostic accuracy.
                      </p>
                    </div>
                  </div>
                )}
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Footer */}
          <div className="flex items-center justify-between px-6 py-4 border-t bg-muted/20">
            <button
              onClick={() => step > 1 && setStep(step - 1)}
              className={`px-4 py-2 rounded-lg text-sm font-medium border transition-colors ${
                step === 1 ? "opacity-40 cursor-not-allowed" : "hover:bg-muted"
              }`}
              disabled={step === 1}
            >
              Previous
            </button>
            <div className="flex items-center gap-1.5">
              {error && (
                <span className="text-[10px] text-destructive font-medium">{error}</span>
              )}
              {steps.map(s => (
                <div key={s.id} className={`w-2 h-2 rounded-full transition-colors ${step >= s.id ? "bg-primary" : "bg-muted-foreground/30"}`} />
              ))}
            </div>
            {step < 4 ? (
              <button
                onClick={() => setStep(step + 1)}
                className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors"
              >
                Next Step
              </button>
            ) : (
              <button
                onClick={handleCreate}
                className="px-4 py-2 rounded-lg bg-primary text-primary-foreground text-sm font-medium hover:bg-primary/90 transition-colors inline-flex items-center gap-1.5"
              >
                <Plus className="w-4 h-4" />
                Create Patient
              </button>
            )}
          </div>
        </motion.div>
      </div>
    </AnimatePresence>
  );
}
