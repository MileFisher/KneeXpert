import { useState, useMemo } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Bell, X, Check, CheckCheck, AlertTriangle, Brain, FileText, Users,
  Clock, Filter, Trash2, Settings
} from "lucide-react";
import { cn } from "@/lib/utils";
import { Popover, PopoverContent, PopoverTrigger } from "@/components/ui/popover";

type NotificationType = "alert" | "scan" | "report" | "system" | "user";

interface Notification {
  id: string;
  type: NotificationType;
  title: string;
  message: string;
  time: string;
  read: boolean;
  urgent?: boolean;
}

const mockNotifications: Notification[] = [
  { id: "n1", type: "alert", title: "Flagged Case", message: "PT-7291 Grade 4 detected — requires urgent review", time: "5 min ago", read: false, urgent: true },
  { id: "n2", type: "scan", title: "Scan Complete", message: "MRI analysis for PT-8842 completed successfully", time: "12 min ago", read: false },
  { id: "n3", type: "report", title: "Report Ready", message: "Diagnostic report for PT-5530 is ready for review", time: "1 hr ago", read: false },
  { id: "n4", type: "system", title: "Model Updated", message: "AI model v3.2.1 deployed with improved accuracy", time: "2 hrs ago", read: true },
  { id: "n5", type: "user", title: "New Referral", message: "Dr. Minh Tuấn referred patient PT-6105 for second opinion", time: "3 hrs ago", read: true },
  { id: "n6", type: "alert", title: "High Pain Alert", message: "PT-3341 reported pain level 9/10 — follow-up needed", time: "4 hrs ago", read: true, urgent: true },
  { id: "n7", type: "scan", title: "Upload Failed", message: "X-Ray upload for PT-2204 failed — retry required", time: "5 hrs ago", read: true },
  { id: "n8", type: "report", title: "Report Exported", message: "PDF report for PT-8842 downloaded by Dr. Thanh Hà", time: "6 hrs ago", read: true },
];

const typeConfig: Record<NotificationType, { icon: typeof Bell; color: string; bg: string }> = {
  alert: { icon: AlertTriangle, color: "text-destructive", bg: "bg-destructive/10" },
  scan: { icon: Brain, color: "text-primary", bg: "bg-primary/10" },
  report: { icon: FileText, color: "text-success", bg: "bg-success/10" },
  system: { icon: Settings, color: "text-muted-foreground", bg: "bg-muted" },
  user: { icon: Users, color: "text-warning", bg: "bg-warning/10" },
};

const filterOptions: { label: string; value: NotificationType | "all" }[] = [
  { label: "All", value: "all" },
  { label: "Alerts", value: "alert" },
  { label: "Scans", value: "scan" },
  { label: "Reports", value: "report" },
  { label: "System", value: "system" },
  { label: "Users", value: "user" },
];

export function NotificationCenter({ collapsed }: { collapsed: boolean }) {
  const [notifications, setNotifications] = useState(mockNotifications);
  const [filter, setFilter] = useState<NotificationType | "all">("all");
  const [showUnreadOnly, setShowUnreadOnly] = useState(false);

  const unreadCount = notifications.filter(n => !n.read).length;

  const filtered = useMemo(() => {
    let list = [...notifications];
    if (filter !== "all") list = list.filter(n => n.type === filter);
    if (showUnreadOnly) list = list.filter(n => !n.read);
    return list;
  }, [notifications, filter, showUnreadOnly]);

  const markAsRead = (id: string) => {
    setNotifications(prev => prev.map(n => n.id === id ? { ...n, read: true } : n));
  };

  const markAllRead = () => {
    setNotifications(prev => prev.map(n => ({ ...n, read: true })));
  };

  const clearAll = () => {
    setNotifications(prev => prev.filter(n => !n.read));
  };

  return (
    <Popover>
      <PopoverTrigger asChild>
        <button
          title={collapsed ? "Notifications" : undefined}
          className={cn(
            "flex items-center gap-3 rounded-lg text-sm transition-colors duration-150 w-full",
            collapsed ? "justify-center px-2 py-2.5" : "px-3 py-2",
            "text-sidebar-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-accent-foreground"
          )}
        >
          <div className="relative flex-shrink-0">
            <Bell className="w-4 h-4" />
            {unreadCount > 0 && (
              <span className="absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full bg-destructive border-2 border-sidebar" />
            )}
          </div>
          {!collapsed && <span className="whitespace-nowrap">Notifications</span>}
          {!collapsed && unreadCount > 0 && (
            <span className="ml-auto text-[10px] font-medium bg-destructive text-destructive-foreground px-1.5 py-0.5 rounded-full">{unreadCount}</span>
          )}
        </button>
      </PopoverTrigger>
      <PopoverContent side="right" align="end" className="w-[380px] p-0 max-h-[520px] flex flex-col">
        {/* Header */}
        <div className="px-4 py-3 border-b flex items-center justify-between flex-shrink-0">
          <div className="flex items-center gap-2">
            <Bell className="w-4 h-4 text-foreground" />
            <h3 className="text-sm font-semibold">Notifications</h3>
            {unreadCount > 0 && (
              <span className="text-[10px] font-medium bg-primary text-primary-foreground px-1.5 py-0.5 rounded-full">{unreadCount} new</span>
            )}
          </div>
          <div className="flex items-center gap-1">
            <button onClick={markAllRead} className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors" title="Mark all read">
              <CheckCheck className="w-3.5 h-3.5" />
            </button>
            <button onClick={clearAll} className="p-1.5 rounded-md text-muted-foreground hover:text-foreground hover:bg-muted transition-colors" title="Clear read">
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        {/* Filters */}
        <div className="px-3 py-2 border-b flex items-center gap-2 flex-shrink-0 overflow-x-auto">
          <Filter className="w-3 h-3 text-muted-foreground flex-shrink-0" />
          {filterOptions.map(f => (
            <button
              key={f.value}
              onClick={() => setFilter(f.value)}
              className={cn(
                "px-2 py-1 rounded-md text-[10px] font-medium transition-all whitespace-nowrap flex-shrink-0",
                filter === f.value ? "bg-primary text-primary-foreground" : "text-muted-foreground hover:bg-muted hover:text-foreground"
              )}
            >
              {f.label}
            </button>
          ))}
          <button
            onClick={() => setShowUnreadOnly(!showUnreadOnly)}
            className={cn(
              "px-2 py-1 rounded-md text-[10px] font-medium transition-all whitespace-nowrap flex-shrink-0 ml-auto",
              showUnreadOnly ? "bg-warning/10 text-warning" : "text-muted-foreground hover:bg-muted"
            )}
          >
            Unread
          </button>
        </div>

        {/* Notification list */}
        <div className="flex-1 overflow-y-auto">
          <AnimatePresence initial={false}>
            {filtered.length > 0 ? filtered.map(n => {
              const config = typeConfig[n.type];
              return (
                <motion.div
                  key={n.id}
                  initial={{ opacity: 0, y: -4 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, height: 0 }}
                  onClick={() => markAsRead(n.id)}
                  className={cn(
                    "px-4 py-3 border-b cursor-pointer transition-colors hover:bg-muted/50",
                    !n.read && "bg-primary/[0.03]"
                  )}
                >
                  <div className="flex items-start gap-3">
                    <div className={cn("w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 mt-0.5", config.bg)}>
                      <config.icon className={cn("w-4 h-4", config.color)} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className={cn("text-xs font-medium", !n.read && "text-foreground")}>{n.title}</span>
                        {n.urgent && <span className="text-[9px] px-1 py-0.5 rounded bg-destructive/10 text-destructive font-medium">Urgent</span>}
                        {!n.read && <span className="w-1.5 h-1.5 rounded-full bg-primary flex-shrink-0 ml-auto" />}
                      </div>
                      <p className="text-[11px] text-muted-foreground mt-0.5 leading-relaxed">{n.message}</p>
                      <div className="flex items-center gap-1 mt-1">
                        <Clock className="w-2.5 h-2.5 text-muted-foreground/60" />
                        <span className="text-[10px] text-muted-foreground/60">{n.time}</span>
                      </div>
                    </div>
                  </div>
                </motion.div>
              );
            }) : (
              <div className="py-12 flex flex-col items-center gap-2 text-muted-foreground">
                <Bell className="w-6 h-6" />
                <p className="text-xs">No notifications</p>
              </div>
            )}
          </AnimatePresence>
        </div>
      </PopoverContent>
    </Popover>
  );
}
