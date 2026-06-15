import { useEffect, useMemo, useRef, useState, forwardRef, useImperativeHandle } from "react";
import { Stage, Layer, Image as KImage, Line, Text as KText, Circle, Group, Transformer, Rect } from "react-konva";
import useImage from "use-image";
import Konva from "konva";
import { Trash2, Type as TypeIcon } from "lucide-react";
import { cn } from "@/lib/utils";

export type EditorTool = "select" | "pan" | "draw" | "text" | "measure" | "annotate";

export interface TextItem {
  id: string;
  x: number;
  y: number;
  text: string;
  fill: string;
  fontSize: number;
  rotation: number;
  width: number;
  scaleX: number;
  scaleY: number;
}
export interface DrawItem {
  id: string;
  points: number[];
  stroke: string;
  strokeWidth: number;
}
export interface MeasureItem {
  id: string;
  x1: number; y1: number; x2: number; y2: number;
}
export interface AnnotationItem {
  id: string;
  x: number; y: number; label: string;
}

export interface KonvaImageEditorHandle {
  exportPNG: (fileName?: string) => string | null;
  clearAll: () => void;
}

interface Props {
  imageUrl: string | null;
  tool: EditorTool;
  brightness: number;
  contrast: number;
  zoom: number;
  drawColor: string;
  drawSize: number;
  textColor: string;
  textFontSize: number;
  measureColor?: string;
  annotateColor?: string;
  onToolChange?: (t: EditorTool) => void;
}

/**
 * KonvaImageEditor: a true canvas editor for radiology scans.
 *
 * - Text boxes are draggable, resizable, rotatable via Konva Transformer.
 * - Placing one text box auto-switches to the "select" tool to prevent mass creation.
 * - Click on empty canvas deselects.
 * - Delete key (or trash icon in floating toolbar) removes the selected text.
 */
export const KonvaImageEditor = forwardRef<KonvaImageEditorHandle, Props>(function KonvaImageEditor(
  { imageUrl, tool, brightness, contrast, zoom, drawColor, drawSize, textColor, textFontSize, measureColor = "#6366f1", annotateColor = "#eab308", onToolChange },
  ref,
) {
  const containerRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<Konva.Stage>(null);
  const transformerRef = useRef<Konva.Transformer>(null);
  const editingInputRef = useRef<HTMLTextAreaElement>(null);

  const [image] = useImage(imageUrl ?? "", "anonymous");
  const [size, setSize] = useState({ width: 800, height: 600 });

  const [drawings, setDrawings] = useState<DrawItem[]>([]);
  const [texts, setTexts] = useState<TextItem[]>([]);
  const [measures, setMeasures] = useState<MeasureItem[]>([]);
  const [measureStart, setMeasureStart] = useState<{ x: number; y: number } | null>(null);
  const [annotations, setAnnotations] = useState<AnnotationItem[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [isDrawing, setIsDrawing] = useState(false);

  // Resize observer
  useEffect(() => {
    if (!containerRef.current) return;
    const obs = new ResizeObserver(entries => {
      const r = entries[0].contentRect;
      setSize({ width: r.width, height: r.height });
    });
    obs.observe(containerRef.current);
    return () => obs.disconnect();
  }, []);

  // Image fit
  const imgGeom = useMemo(() => {
    if (!image) return { x: 0, y: 0, width: size.width, height: size.height };
    const ratio = Math.min(size.width / image.width, size.height / image.height);
    const w = image.width * ratio;
    const h = image.height * ratio;
    return { x: (size.width - w) / 2, y: (size.height - h) / 2, width: w, height: h };
  }, [image, size]);

  // Image filters via CSS-like brightness/contrast on canvas: applied through Konva filters
  useEffect(() => {
    const node = stageRef.current?.findOne(".scan-image") as Konva.Image | undefined;
    if (!node || !image) return;
    node.cache();
    node.filters([Konva.Filters.Brighten, Konva.Filters.Contrast]);
    node.brightness((brightness - 100) / 100);
    node.contrast(contrast - 100);
    node.getLayer()?.batchDraw();
  }, [image, brightness, contrast]);

  // Attach transformer to selected text
  useEffect(() => {
    const tr = transformerRef.current;
    const stage = stageRef.current;
    if (!tr || !stage) return;
    if (selectedId && !editingId) {
      const node = stage.findOne(`#${selectedId}`);
      if (node) {
        tr.nodes([node]);
        tr.getLayer()?.batchDraw();
        return;
      }
    }
    tr.nodes([]);
    tr.getLayer()?.batchDraw();
  }, [selectedId, editingId, texts]);

  // Keyboard delete
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!selectedId || editingId) return;
      if (e.key === "Delete" || e.key === "Backspace") {
        const target = e.target as HTMLElement;
        if (target && (target.tagName === "INPUT" || target.tagName === "TEXTAREA")) return;
        setTexts(prev => prev.filter(t => t.id !== selectedId));
        setSelectedId(null);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [selectedId, editingId]);

  // Stage cursor by tool
  const cursor =
    tool === "draw" ? "crosshair" :
    tool === "text" ? "text" :
    tool === "measure" || tool === "annotate" ? "crosshair" :
    tool === "pan" ? "grab" : "default";

  // ----- Mouse handlers -----
  const getPointer = () => stageRef.current?.getPointerPosition() ?? { x: 0, y: 0 };

  const handleStageMouseDown = (e: Konva.KonvaEventObject<MouseEvent>) => {
    const clickedOnEmpty = e.target === e.target.getStage() || e.target.name() === "scan-image" || e.target.name() === "background";

    if (tool === "select") {
      if (clickedOnEmpty) {
        setSelectedId(null);
        setEditingId(null);
      }
      return;
    }

    const p = getPointer();
    if (tool === "draw") {
      setIsDrawing(true);
      setDrawings(prev => [...prev, { id: `d${Date.now()}`, points: [p.x, p.y], stroke: drawColor, strokeWidth: drawSize }]);
      return;
    }
    if (tool === "text" && clickedOnEmpty) {
      const id = `t${Date.now()}`;
      const newText: TextItem = {
        id, x: p.x, y: p.y, text: "Double-click to edit",
        fill: textColor, fontSize: textFontSize, rotation: 0,
        width: 180, scaleX: 1, scaleY: 1,
      };
      setTexts(prev => [...prev, newText]);
      setSelectedId(id);
      // Auto switch to select to prevent mass creation
      setTimeout(() => onToolChange?.("select"), 0);
      return;
    }
    if (tool === "measure" && clickedOnEmpty) {
      if (!measureStart) {
        setMeasureStart(p);
      } else {
        setMeasures(prev => [...prev, { id: `m${Date.now()}`, x1: measureStart.x, y1: measureStart.y, x2: p.x, y2: p.y }]);
        setMeasureStart(null);
      }
      return;
    }
    if (tool === "annotate" && clickedOnEmpty) {
      const label = `A${annotations.length + 1}`;
      setAnnotations(prev => [...prev, { id: `a${Date.now()}`, x: p.x, y: p.y, label }]);
      return;
    }
  };

  const handleStageMouseMove = () => {
    if (!isDrawing) return;
    const p = getPointer();
    setDrawings(prev => {
      if (prev.length === 0) return prev;
      const last = prev[prev.length - 1];
      const updated = { ...last, points: [...last.points, p.x, p.y] };
      return [...prev.slice(0, -1), updated];
    });
  };

  const handleStageMouseUp = () => {
    if (isDrawing) {
      setIsDrawing(false);
      // Auto-close pen: switch back to select to dismiss the floating color panel
      onToolChange?.("select");
    }
  };

  // Begin editing a text node — render an HTML textarea overlay positioned over the text
  const startEditing = (id: string) => {
    setSelectedId(id);
    setEditingId(id);
  };

  const editingText = texts.find(t => t.id === editingId);
  const editingOverlay = (() => {
    if (!editingText || !stageRef.current) return null;
    const node = stageRef.current.findOne(`#${editingText.id}`) as Konva.Text | undefined;
    if (!node) return null;
    const stageBox = stageRef.current.container().getBoundingClientRect();
    const containerBox = containerRef.current?.getBoundingClientRect();
    const offsetX = containerBox ? stageBox.left - containerBox.left : 0;
    const offsetY = containerBox ? stageBox.top - containerBox.top : 0;
    const abs = node.getAbsolutePosition();
    const scale = node.getAbsoluteScale();
    const rotation = node.rotation();
    const w = (editingText.width ?? 180) * scale.x;
    return (
      <textarea
        ref={editingInputRef}
        autoFocus
        value={editingText.text}
        onChange={e => setTexts(prev => prev.map(t => t.id === editingText.id ? { ...t, text: e.target.value } : t))}
        onBlur={() => setEditingId(null)}
        onKeyDown={e => {
          if (e.key === "Escape") { e.preventDefault(); setEditingId(null); }
        }}
        style={{
          position: "absolute",
          top: abs.y + offsetY,
          left: abs.x + offsetX,
          width: w,
          minHeight: editingText.fontSize * scale.y * 1.4,
          fontSize: editingText.fontSize * scale.y,
          color: editingText.fill,
          transform: `rotate(${rotation}deg)`,
          transformOrigin: "top left",
          background: "rgba(0,0,0,0.45)",
          border: "1px dashed rgba(255,255,255,0.6)",
          outline: "none",
          padding: "2px 4px",
          margin: 0,
          resize: "none",
          fontWeight: 600,
          fontFamily: "inherit",
          lineHeight: 1.2,
          zIndex: 50,
        }}
      />
    );
  })();

  // Floating toolbar near selected text
  const selectedText = texts.find(t => t.id === selectedId);
  const floatingBar = (() => {
    if (!selectedText || editingId || !stageRef.current) return null;
    const node = stageRef.current.findOne(`#${selectedText.id}`) as Konva.Text | undefined;
    if (!node) return null;
    const box = node.getClientRect();
    const containerBox = containerRef.current?.getBoundingClientRect();
    const stageBox = stageRef.current.container().getBoundingClientRect();
    const offsetX = containerBox ? stageBox.left - containerBox.left : 0;
    const offsetY = containerBox ? stageBox.top - containerBox.top : 0;
    return (
      <div
        className="absolute z-40 flex items-center gap-1 bg-background/95 backdrop-blur-sm border rounded-lg px-1.5 py-1 shadow-md"
        style={{ top: Math.max(2, box.y + offsetY - 36), left: box.x + offsetX }}
        onMouseDown={e => e.stopPropagation()}
      >
        <button
          title="Edit text"
          onClick={() => startEditing(selectedText.id)}
          className="w-6 h-6 rounded flex items-center justify-center hover:bg-muted text-muted-foreground"
        >
          <TypeIcon className="w-3.5 h-3.5" />
        </button>
        <button
          title="Delete"
          onClick={() => { setTexts(prev => prev.filter(t => t.id !== selectedText.id)); setSelectedId(null); }}
          className="w-6 h-6 rounded flex items-center justify-center hover:bg-destructive/10 text-destructive"
        >
          <Trash2 className="w-3.5 h-3.5" />
        </button>
      </div>
    );
  })();

  // Imperative API
  useImperativeHandle(ref, () => ({
    exportPNG: (fileName = "annotated.png") => {
      const stage = stageRef.current;
      if (!stage) return null;
      // Hide transformer for export
      transformerRef.current?.visible(false);
      stage.batchDraw();
      const dataURL = stage.toDataURL({ pixelRatio: 2 });
      transformerRef.current?.visible(true);
      stage.batchDraw();
      const link = document.createElement("a");
      link.download = fileName;
      link.href = dataURL;
      link.click();
      return dataURL;
    },
    clearAll: () => {
      setDrawings([]); setTexts([]); setMeasures([]); setAnnotations([]);
      setMeasureStart(null); setSelectedId(null); setEditingId(null);
    },
  }));

  return (
    <div ref={containerRef} className="absolute inset-0 overflow-hidden" style={{ cursor }}>
      <div
        style={{
          width: "100%", height: "100%",
          transform: `scale(${zoom / 100})`,
          transformOrigin: "center center",
          transition: "transform 0.15s ease-out",
        }}
      >
        <Stage
          ref={stageRef}
          width={size.width}
          height={size.height}
          onMouseDown={handleStageMouseDown}
          onMouseMove={handleStageMouseMove}
          onMouseUp={handleStageMouseUp}
          onTouchStart={handleStageMouseDown as any}
          onTouchMove={handleStageMouseMove as any}
          onTouchEnd={handleStageMouseUp as any}
        >
          <Layer>
            <Rect name="background" x={0} y={0} width={size.width} height={size.height} fill="rgba(0,0,0,0)" listening={true} />
            {image && (
              <KImage
                name="scan-image"
                image={image}
                x={imgGeom.x}
                y={imgGeom.y}
                width={imgGeom.width}
                height={imgGeom.height}
                listening={true}
              />
            )}

            {/* Drawings */}
            {drawings.map(d => (
              <Line
                key={d.id}
                points={d.points}
                stroke={d.stroke}
                strokeWidth={d.strokeWidth}
                tension={0.3}
                lineCap="round"
                lineJoin="round"
                listening={false}
              />
            ))}

            {/* Measurements */}
            {measures.map(m => {
              const dist = Math.round(Math.hypot(m.x2 - m.x1, m.y2 - m.y1) * 0.4);
              return (
                <Group key={m.id} listening={tool === "select"}>
                  <Line points={[m.x1, m.y1, m.x2, m.y2]} stroke={measureColor} strokeWidth={2} dash={[6, 4]} />
                  <Circle x={m.x1} y={m.y1} radius={5} fill={measureColor}
                    draggable={tool === "select"}
                    onDragMove={e => setMeasures(prev => prev.map(mm => mm.id === m.id ? { ...mm, x1: e.target.x(), y1: e.target.y() } : mm))}
                  />
                  <Circle x={m.x2} y={m.y2} radius={5} fill={measureColor}
                    draggable={tool === "select"}
                    onDragMove={e => setMeasures(prev => prev.map(mm => mm.id === m.id ? { ...mm, x2: e.target.x(), y2: e.target.y() } : mm))}
                  />
                  <KText x={(m.x1 + m.x2) / 2 - 20} y={(m.y1 + m.y2) / 2 - 18} text={`${dist}mm`} fontSize={12} fill={measureColor} fontStyle="bold" />
                </Group>
              );
            })}
            {measureStart && (
              <Circle x={measureStart.x} y={measureStart.y} radius={4} fill={measureColor} opacity={0.7} />
            )}

            {/* Annotations */}
            {annotations.map(a => (
              <Group
                key={a.id}
                x={a.x} y={a.y}
                draggable={tool === "select"}
                onDragMove={e => setAnnotations(prev => prev.map(aa => aa.id === a.id ? { ...aa, x: e.target.x(), y: e.target.y() } : aa))}
              >
                <Circle radius={11} fill={annotateColor} stroke="#fff" strokeWidth={2} />
                <KText text={a.label} fontSize={10} fontStyle="bold" fill="#000" align="center" verticalAlign="middle" width={22} height={22} offsetX={11} offsetY={11} />
              </Group>
            ))}

            {/* Text boxes */}
            {texts.map(t => (
              <KText
                key={t.id}
                id={t.id}
                x={t.x}
                y={t.y}
                text={t.text}
                fill={t.fill}
                fontSize={t.fontSize}
                fontStyle="600"
                width={t.width}
                rotation={t.rotation}
                scaleX={t.scaleX}
                scaleY={t.scaleY}
                draggable
                onClick={() => { setSelectedId(t.id); }}
                onTap={() => { setSelectedId(t.id); }}
                onDblClick={() => startEditing(t.id)}
                onDblTap={() => startEditing(t.id)}
                onDragEnd={e => {
                  setTexts(prev => prev.map(tt => tt.id === t.id ? { ...tt, x: e.target.x(), y: e.target.y() } : tt));
                }}
                onTransformEnd={e => {
                  const node = e.target as Konva.Text;
                  setTexts(prev => prev.map(tt => tt.id === t.id ? {
                    ...tt,
                    x: node.x(),
                    y: node.y(),
                    rotation: node.rotation(),
                    width: Math.max(40, node.width() * node.scaleX()),
                    scaleX: 1,
                    scaleY: 1,
                  } : tt));
                  node.scaleX(1);
                  node.scaleY(1);
                }}
                visible={editingId !== t.id}
              />
            ))}

            <Transformer
              ref={transformerRef}
              rotateEnabled={true}
              enabledAnchors={["middle-left", "middle-right"]}
              boundBoxFunc={(oldBox, newBox) => {
                if (newBox.width < 30) return oldBox;
                return newBox;
              }}
              borderStroke="#6366f1"
              anchorStroke="#6366f1"
              anchorFill="#fff"
              anchorSize={9}
              rotateAnchorOffset={26}
            />
          </Layer>
        </Stage>
      </div>
      {editingOverlay}
      {floatingBar}
    </div>
  );
});

// Keep style helper
export function bgClassForTool(tool: EditorTool, isDragging: boolean): string {
  return cn("transition-colors", isDragging && "bg-primary/5 ring-2 ring-primary/30 ring-inset");
}