import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes, Outlet } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { ThemeProvider } from "@/components/ThemeProvider";
import { PatientProvider } from "@/context/PatientContext";
import { AppLayout } from "@/components/AppLayout";
import { AdminLayout } from "@/components/AdminLayout";
import DashboardPage from "./pages/DashboardPage";
import PatientsPage from "./pages/PatientsPage";
import DiagnosticsPage from "./pages/DiagnosticsPage";
import ReportsHubPage from "./pages/ReportsHubPage";
import ReportDetailPage from "./pages/ReportDetailPage";
import SettingsPage from "./pages/SettingsPage";
import ProfilePage from "./pages/ProfilePage";
import LoginPage from "./pages/LoginPage";
import SignupPage from "./pages/SignupPage";
import AdminLoginPage from "./pages/AdminLoginPage";
import { AdminOverviewPage, AdminUsersPage, AdminAITrainingPage, AdminDatasetsPage, AdminModelsPage, AdminMRIPipelinePage, AdminSystemPage } from "./pages/AdminPage";
import NotFound from "./pages/NotFound";

const queryClient = new QueryClient();

const App = () => (
  <QueryClientProvider client={queryClient}>
    <ThemeProvider>
      <PatientProvider>
      <TooltipProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter>
          <Routes>
            {/* Auth routes - no layout */}
            <Route path="/login" element={<LoginPage />} />
            <Route path="/signup" element={<SignupPage />} />
            <Route path="/admin/login" element={<AdminLoginPage />} />

            {/* Admin routes - persistent layout */}
            <Route element={<AdminLayout><Outlet /></AdminLayout>}>
              <Route path="/admin" element={<AdminOverviewPage />} />
              <Route path="/admin/users" element={<AdminUsersPage />} />
              <Route path="/admin/ai-training" element={<AdminAITrainingPage />} />
              <Route path="/admin/datasets" element={<AdminDatasetsPage />} />
              <Route path="/admin/models" element={<AdminModelsPage />} />
              <Route path="/admin/mri-pipeline" element={<AdminMRIPipelinePage />} />
              <Route path="/admin/system" element={<AdminSystemPage />} />
            </Route>

            {/* Doctor routes - persistent layout */}
            <Route element={<AppLayout><Outlet /></AppLayout>}>
              <Route path="/" element={<DashboardPage />} />
              <Route path="/patients" element={<PatientsPage />} />
              <Route path="/diagnostics" element={<DiagnosticsPage />} />
              <Route path="/reports" element={<ReportsHubPage />} />
              <Route path="/reports/:patientId" element={<ReportDetailPage />} />
              <Route path="/settings" element={<SettingsPage />} />
              <Route path="/profile" element={<ProfilePage />} />
            </Route>
            <Route path="*" element={<NotFound />} />
          </Routes>
        </BrowserRouter>
      </TooltipProvider>
      </PatientProvider>
    </ThemeProvider>
  </QueryClientProvider>
);

export default App;
