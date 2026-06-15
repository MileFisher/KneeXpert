import { mockPatients, type Patient } from "@/data/patients";

const STORAGE_KEY = "kneexpert_patients_v1";

function hydrateFromMock(stored: Patient[]): Patient[] {
  const storedIds = new Set(stored.map(p => p.id));
  const merged = [...stored];
  for (const seed of mockPatients) {
    if (!storedIds.has(seed.id)) merged.push(seed);
  }
  return merged.map(p => {
    const seed = mockPatients.find(m => m.id === p.id);
    if (!seed) return p;
    return {
      ...seed,
      ...p,
      scans: p.scans?.length ? p.scans : seed.scans,
      timeline: p.timeline?.length ? p.timeline : seed.timeline,
    };
  });
}

export function loadPatients(): Patient[] | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as Patient[];
    if (!Array.isArray(parsed)) return null;
    return hydrateFromMock(parsed);
  } catch {
    return null;
  }
}

export function savePatients(patients: Patient[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(patients));
  } catch (e) {
    console.warn("Failed to persist patients", e);
  }
}

export function clearPatientStorage(): void {
  localStorage.removeItem(STORAGE_KEY);
}
