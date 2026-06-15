import jsPDF from "jspdf";
import "jspdf-autotable";
import type { Patient } from "@/data/patients";

// KL grading criteria
const klCriteria = [
  { grade: 0, label: "Normal", description: "No radiographic features of OA" },
  { grade: 1, label: "Doubtful", description: "Doubtful narrowing of joint space; possible osteophytic lipping" },
  { grade: 2, label: "Minimal", description: "Definite osteophytes; possible narrowing of joint space" },
  { grade: 3, label: "Moderate", description: "Moderate osteophytes; definite narrowing; some sclerosis; possible deformity" },
  { grade: 4, label: "Severe", description: "Large osteophytes; marked narrowing; severe sclerosis; definite deformity" },
];

export function generateReportPDF(patient: Patient): jsPDF {
  const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const pageWidth = doc.internal.pageSize.getWidth();
  const margin = 20;
  const contentWidth = pageWidth - margin * 2;
  let y = margin;

  const currentKL = klCriteria.find(k => k.grade === patient.grade);
  const diagnosisHistory = patient.timeline
    .filter(e => e.type === "diagnosis" && e.grade !== undefined)
    .reverse()
    .map(e => ({ date: e.date, grade: e.grade!, confidence: e.confidence!, summary: e.summary }));

  // Header
  doc.setFillColor(23, 37, 84); // dark blue
  doc.rect(0, 0, pageWidth, 32, "F");
  doc.setTextColor(255, 255, 255);
  doc.setFontSize(18);
  doc.setFont("helvetica", "bold");
  doc.text("KneeXpert", margin, 14);
  doc.setFontSize(9);
  doc.setFont("helvetica", "normal");
  doc.text("Clinical Diagnostic Report", margin, 20);
  doc.text(`Report ID: RPT-${patient.id}-${Date.now().toString(36).toUpperCase()}`, margin, 26);
  doc.text(`Generated: ${new Date().toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" })}`, pageWidth - margin - 60, 26);
  y = 40;

  // Patient Information Box
  doc.setDrawColor(200, 210, 230);
  doc.setFillColor(245, 247, 252);
  doc.roundedRect(margin, y, contentWidth, 30, 2, 2, "FD");
  doc.setTextColor(50, 50, 50);
  doc.setFontSize(11);
  doc.setFont("helvetica", "bold");
  doc.text("Patient Information", margin + 4, y + 7);
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);

  const patientInfo = [
    [`Name: ${patient.name}`, `Age: ${patient.age}`, `Gender: ${patient.gender}`],
    [`ID: ${patient.id}`, `BMI: ${patient.bmi}`, `Pain Level: ${patient.painLevel}/10`],
    [`Last Visit: ${patient.lastVisit}`, `Status: ${patient.status.toUpperCase()}`, `Modality: ${patient.modality.toUpperCase()}`],
  ];
  patientInfo.forEach((row, i) => {
    row.forEach((item, j) => {
      doc.text(item, margin + 4 + j * (contentWidth / 3), y + 14 + i * 5);
    });
  });
  y += 36;

  // Medical History
  doc.setFontSize(11);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(23, 37, 84);
  doc.text("Medical History & Symptoms", margin, y);
  y += 6;
  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(80, 80, 80);
  doc.text(`History: ${patient.history}`, margin, y, { maxWidth: contentWidth });
  y += doc.getTextDimensions(patient.history, { maxWidth: contentWidth }).h + 3;
  doc.text(`Symptoms: ${patient.symptoms}`, margin, y, { maxWidth: contentWidth });
  y += doc.getTextDimensions(patient.symptoms, { maxWidth: contentWidth }).h + 8;

  // AI Diagnostic Summary
  doc.setFillColor(240, 245, 255);
  doc.setDrawColor(100, 140, 220);
  const diagBoxH = 28;
  doc.roundedRect(margin, y, contentWidth, diagBoxH, 2, 2, "FD");
  doc.setFontSize(11);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(23, 37, 84);
  doc.text("AI Diagnostic Summary", margin + 4, y + 7);

  if (patient.grade !== null) {
    doc.setFontSize(20);
    doc.setTextColor(23, 37, 84);
    doc.text(`Grade ${patient.grade}`, margin + 4, y + 20);
    doc.setFontSize(9);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(80, 80, 80);
    doc.text(`${currentKL?.label} Osteoarthritis`, margin + 35, y + 18);
    doc.text(`Confidence: ${patient.aiConfidence}%`, margin + 35, y + 23);

    const modelText = report?.modelUsed
      ?? (patient.modality === "mri" ? "MACS-Net + DeiT-S" : "Ensemble (8 models)");
    doc.text(`Model: ${modelText}`, margin + 90, y + 18);
    doc.text(`Pipeline: ${patient.modality === "xray" ? "Phase I (X-Ray)" : "Phase II (MRI)"}`, margin + 90, y + 23);
  } else {
    doc.setFontSize(9);
    doc.setFont("helvetica", "normal");
    doc.text("Analysis pending. No diagnostic data available.", margin + 4, y + 18);
  }
  y += diagBoxH + 6;

  const report = patient.report;
  if (report?.diagnosisSummary) {
    doc.setFontSize(9);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(60, 60, 60);
    const summaryLines = doc.splitTextToSize(report.diagnosisSummary, contentWidth);
    doc.text(summaryLines, margin, y);
    y += summaryLines.length * 4 + 4;
  }
  if (report?.findings?.length) {
    doc.setFont("helvetica", "bold");
    doc.setTextColor(23, 37, 84);
    doc.text("AI Findings", margin, y);
    y += 5;
    doc.setFont("helvetica", "normal");
    report.findings.forEach(f => {
      const lines = doc.splitTextToSize(`• ${f}`, contentWidth);
      doc.text(lines, margin, y);
      y += lines.length * 4;
    });
    y += 4;
  }

  const imgW = (contentWidth - 6) / 2;
  const imgH = 45;
  if (report?.inputImageDataUrl || report?.ensembleGradcamDataUrl) {
    doc.setFontSize(11);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(23, 37, 84);
    doc.text("Imaging", margin, y);
    y += 6;
    try {
      if (report.inputImageDataUrl?.startsWith("data:image")) {
        doc.addImage(report.inputImageDataUrl, "JPEG", margin, y, imgW, imgH);
        doc.setFontSize(8);
        doc.text("Input scan", margin, y + imgH + 3);
      }
      if (report.ensembleGradcamDataUrl?.startsWith("data:image")) {
        doc.addImage(report.ensembleGradcamDataUrl, "JPEG", margin + imgW + 6, y, imgW, imgH);
        doc.text("Ensemble Grad-CAM", margin + imgW + 6, y + imgH + 3);
      }
      y += imgH + 10;
    } catch {
      y += 4;
    }
  }

  // Clinical Findings
  if (patient.grade !== null) {
    doc.setFontSize(11);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(23, 37, 84);
    doc.text("Clinical Findings", margin, y);
    y += 6;

    const findingsData = [
      ["Joint Space", patient.grade >= 3 ? "Significant narrowing (>50%)" : patient.grade >= 2 ? "Moderate narrowing" : "Mild / Normal"],
      ["Osteophytes", patient.grade >= 3 ? "Definite, multiple" : patient.grade >= 2 ? "Possible formation" : "Doubtful"],
      ["Sclerosis", patient.grade >= 3 ? "Subchondral present" : "Not significant"],
      ["Deformity", patient.grade >= 4 ? "Bone deformity present" : "No significant deformity"],
      ["Cartilage", patient.grade >= 2 ? "Thinning detected" : "Preserved"],
      ["Alignment", patient.grade >= 4 ? "Varus/Valgus deviation" : "Within normal limits"],
    ];

    (doc as any).autoTable({
      startY: y,
      head: [["Finding", "Assessment"]],
      body: findingsData,
      margin: { left: margin, right: margin },
      styles: { fontSize: 8, cellPadding: 2.5, textColor: [60, 60, 60] },
      headStyles: { fillColor: [23, 37, 84], textColor: [255, 255, 255], fontSize: 8 },
      alternateRowStyles: { fillColor: [248, 250, 252] },
      tableWidth: contentWidth,
    });
    y = (doc as any).lastAutoTable.finalY + 6;
  }

  // KL Classification Reference
  if (patient.grade !== null) {
    doc.setFontSize(11);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(23, 37, 84);
    doc.text("Kellgren-Lawrence Classification Reference", margin, y);
    y += 6;

    const klData = klCriteria.map(kl => [
      `Grade ${kl.grade}`,
      kl.label,
      kl.description,
      kl.grade === patient.grade ? "← Current" : "",
    ]);

    (doc as any).autoTable({
      startY: y,
      head: [["Grade", "Label", "Description", ""]],
      body: klData,
      margin: { left: margin, right: margin },
      styles: { fontSize: 7.5, cellPadding: 2, textColor: [60, 60, 60] },
      headStyles: { fillColor: [23, 37, 84], textColor: [255, 255, 255], fontSize: 7.5 },
      columnStyles: { 0: { cellWidth: 15 }, 1: { cellWidth: 18 }, 3: { cellWidth: 18, textColor: [37, 99, 235] } },
      tableWidth: contentWidth,
      didParseCell: (data: any) => {
        if (data.row.index === patient.grade && data.section === "body") {
          data.cell.styles.fillColor = [230, 240, 255];
          data.cell.styles.fontStyle = "bold";
        }
      },
    });
    y = (doc as any).lastAutoTable.finalY + 6;
  }

  // Check if we need a new page
  if (y > 240) {
    doc.addPage();
    y = margin;
  }

  // Diagnosis History
  if (diagnosisHistory.length > 0) {
    doc.setFontSize(11);
    doc.setFont("helvetica", "bold");
    doc.setTextColor(23, 37, 84);
    doc.text("Diagnosis History & Progression", margin, y);
    y += 6;

    const historyData = diagnosisHistory.map(d => [d.date, `Grade ${d.grade}`, `${d.confidence}%`, d.summary]);

    (doc as any).autoTable({
      startY: y,
      head: [["Date", "Grade", "Confidence", "Summary"]],
      body: historyData,
      margin: { left: margin, right: margin },
      styles: { fontSize: 8, cellPadding: 2.5, textColor: [60, 60, 60] },
      headStyles: { fillColor: [23, 37, 84], textColor: [255, 255, 255], fontSize: 8 },
      columnStyles: { 0: { cellWidth: 25 }, 1: { cellWidth: 18 }, 2: { cellWidth: 22 } },
      tableWidth: contentWidth,
    });
    y = (doc as any).lastAutoTable.finalY + 6;
  }

  // Treatment Recommendations
  if (y > 240) { doc.addPage(); y = margin; }

  doc.setFontSize(11);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(23, 37, 84);
  doc.text("Treatment Recommendations", margin, y);
  y += 6;

  doc.setFont("helvetica", "normal");
  doc.setFontSize(9);
  doc.setTextColor(80, 80, 80);

  const recommendations = patient.grade !== null && patient.grade >= 3
    ? [
        "Referral to orthopedic surgery for evaluation of total knee arthroplasty (TKA)",
        "Consider intra-articular corticosteroid or hyaluronic acid injection",
        "Physical therapy: quadriceps strengthening and range of motion exercises",
        `Weight management counseling (current BMI: ${patient.bmi})`,
        "Pain management: NSAIDs, topical analgesics, or tramadol if needed",
        "Follow-up imaging in 3 months to reassess progression",
      ]
    : patient.grade !== null
    ? [
        "Conservative management: NSAIDs and lifestyle modifications",
        "Low-impact exercise: swimming, cycling, walking",
        "Follow-up imaging in 6 months to monitor progression",
        "Physical therapy and joint protection techniques",
        patient.bmi >= 25 ? `Weight loss recommended (current BMI: ${patient.bmi})` : "Maintain healthy weight and activity level",
      ]
    : ["Awaiting diagnosis to generate recommendations."];

  recommendations.forEach(rec => {
    doc.text(`• ${rec}`, margin + 2, y, { maxWidth: contentWidth - 4 });
    y += doc.getTextDimensions(rec, { maxWidth: contentWidth - 4 }).h + 2;
  });
  y += 6;

  // Timeline
  if (y > 220) { doc.addPage(); y = margin; }

  doc.setFontSize(11);
  doc.setFont("helvetica", "bold");
  doc.setTextColor(23, 37, 84);
  doc.text("Complete Patient Timeline", margin, y);
  y += 6;

  const timelineData = patient.timeline.map(e => [
    e.date,
    e.type.charAt(0).toUpperCase() + e.type.slice(1),
    e.summary,
    e.grade !== undefined ? `Grade ${e.grade}` : "",
  ]);

  (doc as any).autoTable({
    startY: y,
    head: [["Date", "Type", "Summary", "Grade"]],
    body: timelineData,
    margin: { left: margin, right: margin },
    styles: { fontSize: 7.5, cellPadding: 2, textColor: [60, 60, 60] },
    headStyles: { fillColor: [23, 37, 84], textColor: [255, 255, 255], fontSize: 7.5 },
    columnStyles: { 0: { cellWidth: 22 }, 1: { cellWidth: 18 }, 3: { cellWidth: 18 } },
    tableWidth: contentWidth,
  });

  // Footer on every page
  const totalPages = doc.getNumberOfPages();
  for (let i = 1; i <= totalPages; i++) {
    doc.setPage(i);
    const pageH = doc.internal.pageSize.getHeight();
    doc.setDrawColor(200, 210, 230);
    doc.line(margin, pageH - 15, pageWidth - margin, pageH - 15);
    doc.setFontSize(7);
    doc.setTextColor(150, 150, 150);
    doc.text("KneeXpert AI Diagnostic System — This report is AI-assisted and should be reviewed by a qualified specialist.", margin, pageH - 10);
    doc.text(`Page ${i} of ${totalPages}`, pageWidth - margin - 20, pageH - 10);
  }

  return doc;
}

export function getReportDataURL(patient: Patient): string {
  const doc = generateReportPDF(patient);
  return doc.output("datauristring");
}

export function downloadReportPDF(patient: Patient): void {
  const doc = generateReportPDF(patient);
  doc.save(`KneeXpert_Report_${patient.id}_${new Date().toISOString().split("T")[0]}.pdf`);
}
