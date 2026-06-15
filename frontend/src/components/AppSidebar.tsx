import { useState } from "react";
import { NavLink, useLocation, useNavigate } from "react-router-dom";
import {
  LayoutDashboard, Users, Scan, FileText, Settings, Activity, User, LogOut,
  PanelLeftClose, PanelLeft
} from "lucide-react";
import { cn } from "@/lib/utils";
import { motion } from "framer-motion";
import { NotificationCenter } from "./NotificationCenter";

const navItems = [
  { icon: LayoutDashboard, label: "Dashboard", path: "/" },
  { icon: Users, label: "Patients", path: "/patients" },
  { icon: Scan, label: "Diagnostics", path: "/diagnostics" },
  { icon: FileText, label: "Reports", path: "/reports" },
  { icon: Settings, label: "Settings", path: "/settings" },
];

function SidebarContent({ collapsed, onNavigate, onToggle }: { collapsed: boolean; onNavigate?: () => void; onToggle?: () => void }) {
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <div className="flex flex-col h-full">
      {/* Logo + Toggle */}
      <div className={cn("h-14 flex items-center border-b border-sidebar-border flex-shrink-0", collapsed ? "justify-center px-2" : "justify-between px-4")}>
        <div className={cn("flex items-center gap-2", collapsed && "justify-center")}>
          <div className="rounded-lg flex items-center justify-center flex-shrink-0">
            <img src="/public/KneeXpert.png" alt="KneeXpert" className="w-10 h-10" />
          </div>
          {!collapsed && (
            <motion.span
              initial={{ opacity: 0, width: 0 }}
              animate={{ opacity: 1, width: "auto" }}
              exit={{ opacity: 0, width: 0 }}
              className="text-sm font-semibold text-foreground whitespace-nowrap overflow-hidden"
            >
              KneeXpert
            </motion.span>
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
        {!collapsed && <span className="section-header px-2 pb-2 block text-sidebar-foreground/60">Workspace</span>}
        {navItems.map((item) => {
          const isActive = location.pathname === item.path ||
            (item.path !== "/" && location.pathname.startsWith(item.path));
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

      {/* Notifications */}
      <div className="px-2 pb-1 flex-shrink-0">
        <NotificationCenter collapsed={collapsed} />
      </div>

      {/* User section */}
      <div className="border-t border-sidebar-border p-3 flex-shrink-0">
        <div className={cn("flex items-center gap-3", collapsed && "justify-center")}>
          <button
            onClick={() => { navigate("/profile"); onNavigate?.(); }}
            className={cn("flex items-center gap-3 flex-1 min-w-0 rounded-lg p-1 -m-1 hover:bg-sidebar-accent/50 transition-colors", collapsed && "justify-center")}
            title="View Profile"
          >
            <div className="rounded-lg flex items-center justify-center flex-shrink-0">
             <img src="/public/KneeXpert.png" alt="KneeXpert" className="w-8 h-8" />
            </div>
            {!collapsed && (
              <div className="flex-1 min-w-0 text-left">
                <p className="text-xs font-medium text-sidebar-accent-foreground truncate">Dr. Quốc Châu</p>
                <p className="text-[10px] text-sidebar-foreground/60 truncate">Radiologist</p>
              </div>
            )}
          </button>
          {!collapsed && (
            <button
              onClick={() => navigate("/login")}
              className="text-sidebar-foreground hover:text-sidebar-accent-foreground transition-colors p-1 rounded-md hover:bg-sidebar-accent/50"
              title="Log out"
            >
              <LogOut className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

export function AppSidebar() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <motion.aside
      initial={false}
      animate={{ width: collapsed ? 64 : 220 }}
      transition={{ duration: 0.2, ease: [0.25, 0.1, 0.25, 1] }}
      className="h-screen flex flex-col bg-sidebar border-r border-sidebar-border sticky top-0 overflow-hidden flex-shrink-0"
    >
      <SidebarContent collapsed={collapsed} onToggle={() => setCollapsed(!collapsed)} />
    </motion.aside>
  );
}

export { SidebarContent, navItems };
