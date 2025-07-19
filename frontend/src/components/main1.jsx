import { useState } from 'react'
import { Routes, Route } from 'react-router-dom'
import Layout from './components/layout/Layout'
import Dashboard from './pages/Dashboard/Dashboard'
import PatientProfile from './pages/PatientProfile/PatientProfile'
import TestResults from './pages/TestResults/TestResults'
import NeuropathyMonitor from './pages/NeuropathyMonitor/NeuropathyMonitor'
import Settings from './pages/Settings/Settings'

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)

  const toggleSidebar = () => {
    setSidebarOpen(prev => !prev)
  }

  return (
    <Layout sidebarOpen={sidebarOpen} toggleSidebar={toggleSidebar}>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/patient-profile" element={<PatientProfile />} />
        <Route path="/test-results" element={<TestResults />} />
        <Route path="/neuropathy-monitor" element={<NeuropathyMonitor />} />
        <Route path="/settings" element={<Settings />} />
      </Routes>
    </Layout>
  )
}

export default App