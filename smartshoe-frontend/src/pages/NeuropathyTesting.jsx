import React, { useState, useEffect } from 'react'
import { Plus, FileText, Users, Activity } from 'lucide-react'
import { useAuth } from '../contexts/AuthContext'
import PatientTestInterface from '../components/neuropathy/PatientTestInterface'
import PhysicianTestResults from '../components/neuropathy/PhysicianTestResults'
import api from '../services/api'
import toast from 'react-hot-toast'

const NeuropathyTesting = () => {
  const { user } = useAuth()
  const [currentView, setCurrentView] = useState('dashboard')
  const [activeTestId, setActiveTestId] = useState(null)
  const [selectedTestId, setSelectedTestId] = useState(null)
  const [patients, setPatients] = useState([])
  const [devices, setDevices] = useState([])
  const [recentTests, setRecentTests] = useState([])
  const [newTestForm, setNewTestForm] = useState({
    patientId: '',
    deviceId: '',
    footSide: 'LEFT',
    isBaseline: false
  })

  useEffect(() => {
    if (user) {
      loadInitialData()
    }
  }, [user])

  const loadInitialData = async () => {
    try {
      // Check if user is authenticated first
      if (!user) {
        console.log('User not authenticated yet, skipping data load')
        return
      }

      const [patientsRes, devicesRes] = await Promise.all([
        api.get('/api/patients'),
        api.get('/api/devices')
      ])
      
      if (patientsRes.data.success) {
        setPatients(patientsRes.data.patients)
      } else if (patientsRes.data.patients) {
        // Handle case where success flag is missing but data exists
        setPatients(patientsRes.data.patients)
      }
      
      if (devicesRes.data.success) {
        setDevices(devicesRes.data.devices)
      } else if (devicesRes.data.devices) {
        // Handle case where success flag is missing but data exists
        setDevices(devicesRes.data.devices)
      }
      
      // Load recent tests for physician view
      if (user?.role !== 'PATIENT') {
        loadRecentTests()
      }
    } catch (error) {
      console.error('Error loading initial data:', error)
      toast.error('Failed to load data. Please check your connection.')
    }
  }

  const loadRecentTests = async () => {
    try {
      // This would need to be implemented in the backend
      // For now, we'll use a placeholder
      setRecentTests([])
    } catch (error) {
      console.error('Error loading recent tests:', error)
    }
  }

  const startNewTest = async () => {
    try {
      const response = await api.post('/api/neuropathy/test/start', newTestForm)
      if (response.data.success) {
        setActiveTestId(response.data.testId)
        setCurrentView('patient-test')
        toast.success('Test started successfully!')
      }
    } catch (error) {
      console.error('Error starting test:', error)
      toast.error('Failed to start test')
    }
  }

  const startPatientTest = async () => {
    try {
      // For patients, automatically create a test with default settings
      if (!user?.username) {
        toast.error('User authentication required')
        return
      }

      // Find an available device (use first active device)
      const availableDevice = devices.find(d => d.status === 'ACTIVE') || devices[0]
      
      if (!availableDevice) {
        toast.error('No device available. Please contact your healthcare provider.')
        return
      }

      // For demo purposes, use the first patient as the default test subject
      // In a real application, this would be mapped to the logged-in user's patient record
      let patientId = 1 // Default to John Doe for demo
      
      // Try to find a patient record that matches the user
      if (user.role === 'PATIENT') {
        // In this demo, just use the first patient
        // In production, you'd have proper user-to-patient mapping
        const defaultPatient = patients.find(p => p.id === 1) // John Doe
        if (defaultPatient) {
          patientId = defaultPatient.id
        }
      }

      const testForm = {
        patientId: patientId,
        deviceId: availableDevice.id,
        footSide: 'LEFT', // Default to left foot
        isBaseline: false
      }

      console.log('Starting test with:', testForm)
      const response = await api.post('/api/neuropathy/test/start', testForm)
      if (response.data.success) {
        setActiveTestId(response.data.testId)
        setCurrentView('patient-test')
        toast.success('Test started successfully!')
      }
    } catch (error) {
      console.error('Error starting patient test:', error)
      console.error('Error details:', error.response?.data)
      toast.error(`Failed to start test: ${error.response?.data?.message || error.message}`)
    }
  }

  const handleTestComplete = (testId) => {
    setActiveTestId(null)
    if (user?.role === 'PATIENT') {
      setCurrentView('dashboard')
      toast.success('Test completed! Your results have been saved.')
    } else {
      setSelectedTestId(testId)
      setCurrentView('physician-results')
    }
  }

  const renderDashboard = () => (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-gray-900">Neuropathy Testing</h1>
        {user?.role !== 'PATIENT' && (
          <button
            onClick={() => setCurrentView('new-test')}
            className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg flex items-center space-x-2"
          >
            <Plus className="w-4 h-4" />
            <span>New Test</span>
          </button>
        )}
      </div>

      {/* Debug Information */}
      <div className="bg-gray-100 p-4 rounded-lg">
        <h3 className="font-medium text-gray-900 mb-2">Debug Information</h3>
        <p className="text-sm text-gray-600">User: {user?.username} ({user?.role})</p>
        <p className="text-sm text-gray-600">Patients loaded: {patients.length}</p>
        <p className="text-sm text-gray-600">Devices loaded: {devices.length}</p>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Total Patients</p>
              <p className="text-2xl font-bold text-gray-900">{patients.length}</p>
            </div>
            <Users className="w-8 h-8 text-blue-600" />
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Active Devices</p>
              <p className="text-2xl font-bold text-gray-900">
                {devices.filter(d => d.status === 'ACTIVE' || d.status === 'LOW_BATTERY').length}
              </p>
            </div>
            <Activity className="w-8 h-8 text-green-600" />
          </div>
        </div>
        
        <div className="bg-white p-6 rounded-lg shadow-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-600">Tests Today</p>
              <p className="text-2xl font-bold text-gray-900">{recentTests.length}</p>
            </div>
            <FileText className="w-8 h-8 text-purple-600" />
          </div>
        </div>
      </div>

      {/* Patient-specific view */}
      {user?.role === 'PATIENT' && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Your Testing</h2>
          <p className="text-gray-600 mb-4">
            Ready to start your neuropathy assessment? The test will evaluate your foot sensation 
            and help monitor your diabetic neuropathy progression.
          </p>
          <button
            onClick={startPatientTest}
            className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium"
          >
            Start Neuropathy Test
          </button>
        </div>
      )}

      {/* Recent Tests Table */}
      {user?.role !== 'PATIENT' && recentTests.length > 0 && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h2 className="text-xl font-bold text-gray-900 mb-4">Recent Tests</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full divide-y divide-gray-200">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Patient</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Date</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Accuracy</th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                {recentTests.map((test) => (
                  <tr key={test.id} className="hover:bg-gray-50">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      {test.patientName}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      {new Date(test.startedAt).toLocaleDateString()}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap">
                      <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
                        test.status === 'COMPLETED' ? 'bg-green-100 text-green-800' :
                        test.status === 'IN_PROGRESS' ? 'bg-blue-100 text-blue-800' :
                        'bg-gray-100 text-gray-800'
                      }`}>
                        {test.status}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                      {test.accuracy ? `${Math.round(test.accuracy * 100)}%` : 'N/A'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-blue-600">
                      <button
                        onClick={() => {
                          setSelectedTestId(test.id)
                          setCurrentView('physician-results')
                        }}
                        className="hover:text-blue-900"
                      >
                        View Results
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )

  const renderNewTestForm = () => (
    <div className="max-w-2xl mx-auto">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-gray-900">Start New Neuropathy Test</h2>
          <button
            onClick={() => setCurrentView('dashboard')}
            className="text-gray-500 hover:text-gray-700"
          >
            ✕
          </button>
        </div>
        
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Patient
            </label>
            <select
              value={newTestForm.patientId}
              onChange={(e) => setNewTestForm(prev => ({ ...prev, patientId: e.target.value }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Choose a patient...</option>
              {patients.map(patient => (
                <option key={patient.id} value={patient.id}>
                  {patient.firstName} {patient.lastName} - {patient.email}
                </option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select Device
            </label>
            <select
              value={newTestForm.deviceId}
              onChange={(e) => setNewTestForm(prev => ({ ...prev, deviceId: e.target.value }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Choose a device...</option>
              {devices.filter(device => device.status === 'ACTIVE' || device.status === 'LOW_BATTERY').map(device => (
                <option key={device.id} value={device.id}>
                  {device.model} - {device.serialNumber}
                </option>
              ))}
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Foot Side
            </label>
            <select
              value={newTestForm.footSide}
              onChange={(e) => setNewTestForm(prev => ({ ...prev, footSide: e.target.value }))}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="LEFT">Left Foot</option>
              <option value="RIGHT">Right Foot</option>
              <option value="BOTH">Both Feet</option>
            </select>
          </div>
          
          <div className="flex items-center">
            <input
              type="checkbox"
              id="isBaseline"
              checked={newTestForm.isBaseline}
              onChange={(e) => setNewTestForm(prev => ({ ...prev, isBaseline: e.target.checked }))}
              className="mr-2"
            />
            <label htmlFor="isBaseline" className="text-sm text-gray-700">
              This is a baseline test (first test for this patient)
            </label>
          </div>
          
          <div className="flex space-x-4 pt-4">
            <button
              onClick={() => setCurrentView('dashboard')}
              className="flex-1 bg-gray-200 text-gray-700 py-2 px-4 rounded-lg hover:bg-gray-300"
            >
              Cancel
            </button>
            <button
              onClick={startNewTest}
              disabled={!newTestForm.patientId || !newTestForm.deviceId}
              className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white py-2 px-4 rounded-lg"
            >
              Start Test
            </button>
          </div>
        </div>
      </div>
    </div>
  )

  const renderCurrentView = () => {
    switch (currentView) {
      case 'patient-test':
        return (
          <PatientTestInterface
            testId={activeTestId}
            onTestComplete={handleTestComplete}
          />
        )
      case 'physician-results':
        return selectedTestId ? (
          <div>
            <div className="mb-4">
              <button
                onClick={() => setCurrentView('dashboard')}
                className="text-blue-600 hover:text-blue-800"
              >
                ← Back to Dashboard
              </button>
            </div>
            <PhysicianTestResults testId={selectedTestId} />
          </div>
        ) : null
      case 'new-test':
        return renderNewTestForm()
      default:
        return renderDashboard()
    }
  }

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-7xl mx-auto">
        {renderCurrentView()}
      </div>
    </div>
  )
}

export default NeuropathyTesting