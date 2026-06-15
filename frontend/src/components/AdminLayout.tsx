import { ReactNode, useState } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import {
  Activity, LayoutDashboard, Users, Brain, Database, Cpu, Layers, Settings,
  User, LogOut, Menu, PanelLeftClose, PanelLeft
} from "lucide-react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { useIsMobile } from "@/hooks/use-mobile";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";

const adminNavItems = [
  { icon: LayoutDashboard, label: "Overview", path: "/admin" },
  { icon: Users, label: "User Management", path: "/admin/users" },
  { icon: Brain, label: "AI Training", path: "/admin/ai-training" },
  { icon: Database, label: "Datasets", path: "/admin/datasets" },
  { icon: Cpu, label: "Models", path: "/admin/models" },
  { icon: Layers, label: "MRI Pipeline", path: "/admin/mri-pipeline" },
  { icon: Settings, label: "System", path: "/admin/system" },
];

function AdminSidebarContent({ collapsed, onNavigate, onToggle }: { collapsed: boolean; onNavigate?: () => void; onToggle?: () => void }) {
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <div className="flex flex-col h-full">
      {/* Logo + Toggle */}
      <div className={cn("h-14 flex items-center border-b border-sidebar-border flex-shrink-0", collapsed ? "justify-center px-2" : "justify-between px-4")}>
        <div className={cn("flex items-center gap-3", collapsed && "justify-center")}>
          <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center flex-shrink-0">
            <Activity className="w-4 h-4 text-primary-foreground" />
          </div>
          {!collapsed && (
            <div className="flex flex-col">
              <span className="text-sm font-semibold text-sidebar-accent-foreground whitespace-nowrap">KneeXpert</span>
              <span className="text-[10px] text-primary font-medium -mt-0.5">Admin</span>
            </div>
          )}
        </div>
        {!collapsed && onToggle && (
          <button
            onClick={onToggle}
            className="p-1.5 rounded-md text-sidebar-foreground/60 hover:text-sidebar-accent-foreground hover:bg-sidebar-accent/50 transition-colors"
            title="Collapse sidebar"
          >
            <PanelLeftClose className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Expand button when collapsed */}
      {collapsed && onToggle && (
        <div className="flex justify-center pt-2 px-2">
          <button
            onClick={onToggle}
            className="p-2 rounded-md text-sidebar-foreground/60 hover:text-sidebar-accent-foreground hover:bg-sidebar-accent/50 transition-colors"
            title="Expand sidebar"
          >
            <PanelLeft className="w-4 h-4" />
          </button>
        </div>
      )}

      {/* Navigation */}
      <nav className="flex-1 py-3 px-2 space-y-0.5 overflow-y-auto">
        {!collapsed && <span className="section-header px-2 pb-2 block text-sidebar-foreground/60">Management</span>}
        {adminNavItems.map(item => {
          const isActive = location.pathname === item.path;
          return (
            <NavLink
              key={item.path}
              to={item.path}
              onClick={onNavigate}
              title={collapsed ? item.label : undefined}
              className={cn(
                "flex items-center gap-3 rounded-lg text-sm transition-colors duration-150",
                collapsed ? "justify-center px-2 py-2.5" : "px-3 py-2",
                isActive
                  ? "bg-sidebar-accent text-sidebar-accent-foreground font-semibold"
                  : "text-sidebar-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-accent-foreground"
              )}
            >
              <item.icon className={cn("w-4 h-4 flex-shrink-0", isActive && "text-sidebar-accent-foreground")} />
              {!collapsed && <span className="whitespace-nowrap">{item.label}</span>}
            </NavLink>
          );
        })}
      </nav>

      {/* Switch to Doctor View */}
      <div className="px-2 pb-1 flex-shrink-0">
        <button
          onClick={() => { onNavigate?.(); navigate("/"); }}
          title={collapsed ? "Doctor Portal" : undefined}
          className={cn(
            "flex items-center gap-3 rounded-lg text-sm transition-colors duration-150 w-full",
            collapsed ? "justify-center px-2 py-2.5" : "px-3 py-2",
            "text-sidebar-foreground hover:bg-sidebar-accent/50 hover:text-sidebar-accent-foreground"
          )}
        >
          <Activity className="w-4 h-4 flex-shrink-0" />
          {!collapsed && <span className="whitespace-nowrap text-xs">Doctor Portal →</span>}
        </button>
      </div>

      {/* User section */}
      <div className="border-t border-sidebar-border p-3 flex-shrink-0">
        <div className={cn("flex items-center gap-3", collapsed && "justify-center")}>
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-primary/80 to-primary flex items-center justify-center flex-shrink-0">
            <User className="w-4 h-4 text-primary-foreground" />
          </div>
          {!collapsed && (
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium text-sidebar-accent-foreground truncate">System Admin</p>
              <p className="text-[10px] text-sidebar-foreground/60 truncate">admin@kneexpert.com</p>
            </div>
          )}
          {!collapsed && (
            <button
              onClick={() => navigate("/admin/login")}
              className="text-sidebar-foreground hover:text-sidebar-accent-foreground transition-colors p-1 rounded-md hover:bg-sidebar-accent/50"
            >
              <LogOut className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export function AdminLayout({ children }: { children: ReactNode }) {
  const [collapsed, setCollapsed] = useState(false);
  const isMobile = useIsMobile();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="min-h-screen flex w-full bg-background">
      {!isMobile && (
        <motion.aside
          initial={false}
          animate={{ width: collapsed ? 64 : 240 }}
          transition={{ duration: 0.2, ease: [0.25, 0.1, 0.25, 1] }}
          className="h-screen flex flex-col bg-sidebar border-r border-sidebar-border sticky top-0 overflow-hidden flex-shrink-0"
        >
          <AdminSidebarContent collapsed={collapsed} onToggle={() => setCollapsed(!collapsed)} />
        </motion.aside>
      )}
      <div className="flex-1 min-w-0 flex flex-col">
        {isMobile && (
          <header className="h-12 flex items-center px-4 gap-3 border-b bg-background sticky top-0 z-40">
            <button onClick={() => setMobileOpen(true)} className="p-1.5 rounded-md hover:bg-muted transition-colors">
              <Menu className="w-5 h-5 text-foreground" />
            </button>
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-md bg-primary flex items-center justify-center">
                <Activity className="w-3 h-3 text-primary-foreground" />
              </div>
              <span className="text-sm font-semibold">KneeXpert</span>
              <span className="text-[10px] text-primary font-medium">Admin</span>
            </div>
          </header>
        )}
        <main className="flex-1 min-w-0 overflow-auto">{children}</main>
      </div>
      {isMobile && (
        <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
          <SheetContent side="left" className="w-[260px] p-0 bg-sidebar border-sidebar-border">
            <SheetTitle className="sr-only">Admin Navigation</SheetTitle>
            <AdminSidebarContent collapsed={false} onNavigate={() => setMobileOpen(false)} />
          </SheetContent>
        </Sheet>
      )}
    </div>
  );
}
