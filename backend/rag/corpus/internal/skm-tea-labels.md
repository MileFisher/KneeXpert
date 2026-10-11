# SKM-TEA diagnostic categories

Definitions for the sixteen multi-label categories the KneeXpert MRI pipeline predicts,
matching `backend/mri/categories.json` exactly. Written for this project because the
label names are dataset shorthand: published literature describes these pathologies at
length but never under these exact names, so a corpus of research articles alone cannot
ground a statement about what a given predicted label means.

Categories are not mutually exclusive. A single study commonly carries several, and the
SKM-TEA annotations average roughly 2.4 labels per study.

## Meniscal tear patterns

The meniscus is a fibrocartilaginous structure distributing load across the
tibiofemoral joint. Tear morphology determines whether a tear is mechanically
significant and whether it is repairable.

**Meniscal Tear (Myxoid)** — myxoid degeneration signal within the meniscal substance,
without a discrete tear plane reaching a surface. Generally a degenerative finding
rather than an acute injury.

**Meniscal Tear (Horizontal)** — a horizontal cleavage plane splitting the meniscus
into upper and lower leaves. Typically degenerative.

**Meniscal Tear (Radial)** — a tear running perpendicular to the circumferential
fibres, extending toward or reaching the capsular junction. Because it severs the
load-distributing fibres, a radial tear compromises hoop-stress function
disproportionately to its size.

**Meniscal Tear (Vertical/Longitudinal)** — a tear running along the circumferential
fibres. Preserves hoop stress better than a radial tear and is more often repairable.

**Meniscal Tear (Oblique)** — an oblique tear plane, sometimes described as a
parrot-beak configuration, with features of both radial and longitudinal patterns.

**Meniscal Tear (Complex)** — tearing in multiple planes. Associated with a higher risk
of mechanical symptoms and less amenable to repair.

**Meniscal Tear (Flap)** — a tear producing a displaced fragment. The displaced flap can
cause catching or locking.

**Meniscal Tear (Extrusion)** — displacement of meniscal tissue beyond the tibial
plateau margin. Indicates root detachment or radial deficiency and is strongly
associated with progression of cartilage loss.

## Ligament injury grades

Graded by the degree of fibre disruption, which determines whether the ligament remains
functionally competent.

**Ligament Tear (Low-Grade Sprain)** — fibres intact with mild abnormal signal.
Functionally competent.

**Ligament Tear (Moderate Grade Sprain or Mucoid Degeneration)** — partial fibre
disruption, or mucoid degeneration of the ligament substance. Partially competent.

**Ligament Tear (Full Thickness/Complete Tear)** — complete discontinuity of fibres.
The ligament is functionally incompetent, with implications for joint stability.

## Cartilage lesion grades

A depth-based scale. Depth is the primary determinant of management, because lesions
reaching subchondral bone do not heal spontaneously.

**Cartilage Lesion (1)** — signal change with an intact articular surface, corresponding
to softening or fibrillation.

**Cartilage Lesion (2A)** — superficial partial-thickness defect, extending into the
superficial zone only.

**Cartilage Lesion (2B)** — deep partial-thickness defect, extending into the deep zone
but not reaching subchondral bone.

**Cartilage Lesion (3)** — full-thickness defect with subchondral bone exposed.

## Joint effusion

**Effusion** — excess synovial fluid within the joint capsule. A non-specific marker of
intra-articular pathology or inflammation. Effusion in isolation does not localise a
cause, but its presence raises the prior probability that some intra-articular
abnormality is present, and its absence does not exclude one.

## Relationship to radiographic grading

These categories describe soft tissue and are therefore largely invisible on plain
radiography, which is precisely the diagnostic gap MRI fills. A knee with a normal
Kellgren-Lawrence grade can carry a full-thickness cartilage defect or a complete
ligament tear. Conversely, advanced radiographic osteoarthritis can coexist with, and
visually mask, concurrent soft-tissue pathology. A mapping from these categories to a
Kellgren-Lawrence grade is therefore an approximation across different tissue types and
should not be read as a radiographic measurement.
