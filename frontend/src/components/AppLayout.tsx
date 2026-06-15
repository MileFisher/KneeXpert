import { ReactNode, useState } from "react";
import { AppSidebar, SidebarContent } from "./AppSidebar";
import { useIsMobile } from "@/hooks/use-mobile";
import { Menu, Activity } from "lucide-react";
import { Sheet, SheetContent, SheetTitle } from "@/components/ui/sheet";
import { TutorialOverlay } from "./TutorialOverlay";

interface AppLayoutProps {
  children: ReactNode;
}

export function AppLayout({ children }: AppLayoutProps) {
  const isMobile = useIsMobile();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <div className="min-h-screen flex w-full bg-background">
      <TutorialOverlay />
      {!isMobile && <AppSidebar />}
      <div className="flex-1 min-w-0 flex flex-col">
        {isMobile && (
          <header className="h-12 flex items-center px-4 gap-3 border-b bg-background sticky top-0 z-40">
            <button
              onClick={() => setMobileOpen(true)}
              className="p-1.5 rounded-md hover:bg-muted transition-colors"
            >
              <Menu className="w-5 h-5 text-foreground" />
            </button>
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-md bg-primary flex items-center justify-center">
                <Activity className="w-3 h-3 text-primary-foreground" />
              </div>
              <span className="text-sm font-semibold">KneeXpert</span>
            </div>
          </header>
        )}
        <main className="flex-1 min-w-0 overflow-auto flex flex-col">
          {children}
        </main>
      </div>
      {isMobile && (
        <Sheet open={mobileOpen} onOpenChange={setMobileOpen}>
          <SheetContent side="left" className="w-[260px] p-0 bg-sidebar border-sidebar-border">
            <SheetTitle className="sr-only">Navigation</SheetTitle>
            <SidebarContent collapsed={false} onNavigate={() => setMobileOpen(false)} />
          </SheetContent>
        </Sheet>
      )}
    </div>
  );
}
