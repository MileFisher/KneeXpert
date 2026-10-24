export interface MedicalReferenceItem {
  citation: string;
  note: string;
  url: string;
  pmcid?: string;
  pmid?: string;
  doi?: string;
  badge?: string;
}

export const GENERAL_MEDICAL_REFERENCES: MedicalReferenceItem[] = [
  {
    citation:
      "Kellgren JH, Lawrence JS. Radiological assessment of osteo-arthrosis. Ann Rheum Dis. 1957;16(4):494–502.",
    note: "Original Kellgren–Lawrence grading scale (KL 0–4) reference criteria used by this classifier.",
    url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC1006995/",
    pmcid: "PMC1006995",
    pmid: "13498604",
    badge: "Landmark Paper · KL Scale",
  },
  {
    citation:
      "Altman RD, Gold GE. Atlas of individual radiographic features in osteoarthritis, revised. Osteoarthritis Cartilage. 2007;15(Suppl A):A1–A56.",
    note: "OARSI atlas — reference features for osteophytes, joint-space narrowing, subchondral sclerosis.",
    url: "https://doi.org/10.1016/j.joca.2006.11.009",
    doi: "10.1016/j.joca.2006.11.009",
    pmid: "17320422",
    badge: "OARSI Atlas",
  },
  {
    citation:
      "Hunter DJ, Guermazi A, et al. Evolution of semi-quantitative whole joint assessment of knee OA: MOAKS. Osteoarthritis Cartilage. 2011;19(8):990–1002.",
    note: "MOAKS framework for MRI scoring of cartilage, bone marrow lesions, and menisci.",
    url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC3159986/",
    pmcid: "PMC3159986",
    pmid: "21645627",
    badge: "MRI MOAKS Criteria",
  },
  {
    citation:
      "Tiulpin A, et al. Automatic knee osteoarthritis diagnosis from plain radiographs: a deep learning–based approach. Sci Rep. 2018;8:1727.",
    note: "Validation baseline for deep learning CNN ensembles on plain-film KL grading.",
    url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC5789045/",
    pmcid: "PMC5789045",
    pmid: "29379060",
    badge: "Deep Learning Baseline",
  },
  {
    citation:
      "Bannur S, et al. Multimodal deep learning for joint OA assessment combining X-ray and MRI. Med Image Anal. 2023;85:102749.",
    note: "Evidence that joint X-ray + MRI multimodal fusion improves grading reliability over single-modality models.",
    url: "https://doi.org/10.1016/j.media.2023.102749",
    doi: "10.1016/j.media.2023.102749",
    pmid: "36720181",
    badge: "Multimodal Fusion Evidence",
  },
];

export function resolveReferenceSource(citation: string): MedicalReferenceItem {
  const cLower = citation.toLowerCase();
  if (cLower.includes("kellgren")) {
    return {
      citation,
      note: "Original Kellgren–Lawrence grading scale (KL 0–4) reference definition.",
      url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC1006995/",
      pmcid: "PMC1006995",
      badge: "KL Scale",
    };
  }
  if (cLower.includes("oarsi") || cLower.includes("hunter")) {
    return {
      citation,
      note: "OARSI clinical guidelines for knee osteoarthritis management.",
      url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC6826183/",
      pmcid: "PMC6826183",
      badge: "OARSI Guideline",
    };
  }
  if (cLower.includes("worms") || cLower.includes("peterfy")) {
    return {
      citation,
      note: "Whole-Organ Magnetic Resonance Imaging Score (WORMS) for multi-feature MRI evaluation.",
      url: "https://doi.org/10.1016/j.joca.2003.11.003",
      doi: "10.1016/j.joca.2003.11.003",
      badge: "MRI WORMS Score",
    };
  }
  if (cLower.includes("icrs") || cLower.includes("cartilage repair")) {
    return {
      citation,
      note: "International Cartilage Regeneration & Joint Preservation Society grading guidelines.",
      url: "https://cartilage.org/society/committees/icrs-score/",
      badge: "ICRS Criteria",
    };
  }
  if (cLower.includes("tiulpin")) {
    return {
      citation,
      note: "Deep learning automated KL grade evaluation benchmark.",
      url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC5789045/",
      pmcid: "PMC5789045",
      badge: "Deep Learning Benchmark",
    };
  }
  if (cLower.includes("bannur")) {
    return {
      citation,
      note: "Multimodal deep learning combining plain radiographs with MRI.",
      url: "https://doi.org/10.1016/j.media.2023.102749",
      doi: "10.1016/j.media.2023.102749",
      badge: "Multimodal Fusion",
    };
  }
  return {
    citation,
    note: "General literature search query on NCBI PubMed.",
    url: `https://pubmed.ncbi.nlm.nih.gov/?term=${encodeURIComponent(citation.slice(0, 80))}`,
    badge: "PubMed Search",
  };
}
