import React, { useState, useEffect, useRef } from 'react'
import { Clock, Activity, CheckCircle, AlertCircle } from 'lucide-react'
import FootDiagram from './FootDiagram'
import { TestSimulationService } from './TestSimulationService'
import RealTimeRiskAssessment from './RealTimeRiskAssessment'
import api from '../../services/api'
import toast from 'react-hot-toast'
import { useWebSocket } from '../../contexts/WebSocketContext'
import { useNotifications } from '../../contexts/NotificationContext'

const PatientTestInterface = ({ testId, onTestComplete }) => {
  const [testStatus, setTestStatus] = useState('loading')
  const [currentStimulus, setCurrentStimulus] = useState(null)
  const [showInstructions, setShowInstructions] = useState(true)
  const [testData, setTestData] = useState(null)
  const [timeElapsed, setTimeElapsed] = useState(0)
  const [response, setResponse] = useState({
    feltSensation: false,
    perceivedIntensity: 5,
    perceivedType: '',
    perceivedLocation: null,
    responseConfidence: 3
  })
  const [footView, setFootView] = useState('top')
  const [simulationRunning, setSimulationRunning] = useState(false)
  const [simulationProgress, setSimulationProgress] = useState({ current: 0, total: 0, percentage: 0 })
  const [awaitingResponse, setAwaitingResponse] = useState(false)
  const [currentRiskLevel, setCurrentRiskLevel] = useState(null)
  const simulationService = useRef(new TestSimulationService())
  const { isConnected } = useWebSocket()
  const { showSuccess, showError } = useNotifications()

  useEffect(() => {
    let timer;
    
    if (testId) {
      loadTestStatus()
    }
    
    timer = setInterval(() => {
      setTimeElapsed(prev => prev + 1)
    }, 1000)
    
    // Listen for real-time test results
    const handleTestResult = (event) => {
      const data = event.detail
      if (data.testId === testId) {
        setTestStatus('completed')
        showSuccess('Test completed successfully!')
        if (onTestComplete) {
          onTestComplete(data)
        }
      }
    }

    const handleDeviceData = (event) => {
      const data = event.detail
      if (data.testId === testId) {
        setCurrentStimulus(data)
      }
    }

    window.addEventListener('testResultReceived', handleTestResult)
    window.addEventListener('deviceDataReceived', handleDeviceData)
    
    return () => {
      if (timer) {
        clearInterval(timer)
      }
      // Cleanup simulation service
      if (simulationService.current?.isRunning) {
        simulationService.current.stopTest()
      }
      window.removeEventListener('testResultReceived', handleTestResult)
      window.removeEventListener('deviceDataReceived', handleDeviceData)
    }
  }, [testId, onTestComplete, showSuccess])

  const loadTestStatus = async () => {
    if (!testId) {
      console.log('No test ID provided')
      setTestStatus('error')
      return
    }
    
    try {
      const response = await api.get(`/api/neuropathy/test/${testId}/status`)
      if (response && response.data && response.data.success) {
        setTestData(response.data)
        setTestStatus(response.data.status ? response.data.status.toLowerCase() : 'error')
        
        if (response.data.status === 'PENDING') {
          setShowInstructions(true)
        }
      } else {
        console.error('Invalid response format:', response)
        setTestStatus('error')
        toast.error('Invalid response from server')
      }
    } catch (error) {
      console.error('Error loading test status:', error)
      setTestStatus('error')
      if (error.response) {
        toast.error(`Failed to load test status: ${error.response.status} ${error.response.statusText}`)
      } else if (error.request) {
        toast.error('Network error: Unable to connect to server')
      } else {
        toast.error('Failed to load test status')
      }
    }
  }

  const beginTest = async () => {
    try {
      // Start the backend test
      const response = await api.post(`/api/neuropathy/test/${testId}/begin`)
      if (response.data.success) {
        setShowInstructions(false)
        setTestStatus('in_progress')
        
        // Initialize and start the simulation service
        const service = simulationService.current
        service.generateTestSequence(20) // Generate 20 stimuli
        
        // Set up simulation callbacks
        const onStimulusGenerated = (stimulus) => {
          console.log('New stimulus presented:', stimulus)
          setCurrentStimulus(stimulus)
          setAwaitingResponse(true)
          resetResponse()
          
          // Update progress
          const progress = service.getProgress()
          setSimulationProgress(progress)
          setTestData(prev => ({
            ...prev,
            progress: progress.percentage,
            completedStimuli: progress.current,
            totalStimuli: progress.total
          }))
        }
        
        const onSimulationComplete = () => {
          console.log('Simulation test completed')
          setSimulationRunning(false)
          setAwaitingResponse(false)
          setTestStatus('completed')
          toast.success('Test completed!')
          onTestComplete && onTestComplete(testId)
        }
        
        // Start the simulation
        service.startTest(onStimulusGenerated, onSimulationComplete)
        setSimulationRunning(true)
        
        toast.success('Test started!')
      }
    } catch (error) {
      console.error('Error beginning test:', error)
      toast.error('Failed to start test')
    }
  }

  const submitResponse = async () => {
    if (!currentStimulus || !awaitingResponse) return

    try {
      setAwaitingResponse(false)
      
      // Submit response to simulation service
      const service = simulationService.current
      const responseData = {
        feltSensation: response.feltSensation,
        perceivedIntensity: response.perceivedIntensity,
        perceivedType: response.perceivedType,
        perceivedLocation: response.perceivedLocation,
        responseConfidence: response.responseConfidence
      }
      
      // Let simulation service handle progression
      const responseSubmitted = service.submitResponse(responseData)
      
      if (responseSubmitted) {
        // Also submit to backend API for data storage
        try {
          const backendResponse = {
            stimulusId: currentStimulus.id,
            feltSensation: response.feltSensation,
            ...(response.feltSensation && {
              perceivedIntensity: response.perceivedIntensity,
              perceivedType: response.perceivedType,
              perceivedLocation: JSON.stringify(response.perceivedLocation),
              responseConfidence: response.responseConfidence
            })
          }
          
          // Submit to backend (don't block on this)
          api.post(`/api/neuropathy/test/${testId}/respond`, backendResponse)
            .catch(error => console.warn('Backend submit failed:', error))
            
        } catch (error) {
          console.warn('Backend submission error:', error)
        }
        
        toast.success('Response recorded')
      } else {
        console.warn('Failed to submit response to simulation service')
        setAwaitingResponse(true) // Allow retry
      }
    } catch (error) {
      console.error('Error submitting response:', error)
      toast.error('Failed to submit response')
      setAwaitingResponse(true) // Allow retry
    }
  }

  const resetResponse = () => {
    setResponse({
      feltSensation: false,
      perceivedIntensity: 5,
      perceivedType: '',
      perceivedLocation: null,
      responseConfidence: 3
    })
  }

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  const getStimulusTypeOptions = () => [
    { value: 'VIBRATION', label: 'Vibration' },
    { value: 'TEMPERATURE_HOT', label: 'Hot Temperature' },
    { value: 'TEMPERATURE_COLD', label: 'Cold Temperature' },
    { value: 'PINPRICK', label: 'Sharp/Pinprick' }
  ]

  const getStimulusIcon = (type) => {
    switch (type) {
      case 'VIBRATION':
        return <div className="text-3xl animate-bounce">📳</div>
      case 'TEMPERATURE_HOT':
        return <div className="text-3xl text-red-500">🔥</div>
      case 'TEMPERATURE_COLD':
        return <div className="text-3xl text-blue-500">❄️</div>
      case 'PINPRICK':
        return <div className="text-3xl text-yellow-600">📌</div>
      default:
        return <div className="text-3xl">🎯</div>
    }
  }

  const getStimulusMessage = (type) => {
    switch (type) {
      case 'VIBRATION':
        return 'Vibration stimulus active'
      case 'TEMPERATURE_HOT':
        return 'Hot temperature stimulus active'
      case 'TEMPERATURE_COLD':
        return 'Cold temperature stimulus active'
      case 'PINPRICK':
        return 'Sharp sensation stimulus active'
      default:
        return 'Stimulus active'
    }
  }

  if (testStatus === 'loading') {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2 text-gray-600">Loading test...</span>
      </div>
    )
  }

  if (testStatus === 'error' || !testId) {
    return (
      <div className="max-w-lg mx-auto p-6 bg-white rounded-lg shadow-lg text-center">
        <AlertCircle className="w-16 h-16 text-red-600 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Test Not Available</h2>
        <p className="text-gray-600 mb-4">
          There was an issue loading the test. Please try starting a new test.
        </p>
        <button
          onClick={() => window.history.back()}
          className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3 rounded-lg font-medium"
        >
          Go Back
        </button>
      </div>
    )
  }

  if (showInstructions && testStatus === 'pending') {
    return (
      <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-lg">
        <div className="text-center mb-6">
          <AlertCircle className="w-12 h-12 text-blue-600 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-gray-900 mb-4">Neuropathy Test Instructions</h2>
        </div>
        
        <div className="space-y-4 text-gray-700 mb-8">
          <p className="text-lg leading-relaxed">
            Welcome to the neuropathy test. During this test, you may feel different sensations on your foot including:
          </p>
          
          <ul className="list-disc list-inside space-y-2 ml-4">
            <li><strong>Vibration</strong> - A buzzing or tingling sensation</li>
            <li><strong>Temperature</strong> - Hot or cold sensations</li>
            <li><strong>Pinprick</strong> - Sharp but safe sensations</li>
          </ul>
          
          <div className="bg-yellow-50 border-l-4 border-yellow-400 p-4 my-6">
            <div className="flex">
              <AlertCircle className="w-5 h-5 text-yellow-400 mt-0.5 mr-3" />
              <div>
                <h4 className="text-sm font-medium text-yellow-800">Important Note</h4>
                <p className="text-sm text-yellow-700">
                  Sometimes, <strong>no stimulus will be present</strong>. It is completely normal to not feel anything during some trials. 
                  Please respond honestly about what you feel or don't feel.
                </p>
              </div>
            </div>
          </div>
          
          <p>
            For each sensation you feel, you'll be asked to:
          </p>
          <ul className="list-disc list-inside space-y-1 ml-4">
            <li>Indicate if you felt something</li>
            <li>Rate the intensity (1-10 scale)</li>
            <li>Identify the type of sensation</li>
            <li>Show where you felt it on the foot diagram</li>
            <li>Rate your confidence in the response</li>
          </ul>
          
          <p className="text-sm text-gray-600">
            Estimated duration: {testData?.estimatedDuration || 15} minutes
          </p>
        </div>
        
        <div className="text-center">
          <button
            onClick={beginTest}
            className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-lg font-medium transition-colors"
          >
            Start Test
          </button>
        </div>
      </div>
    )
  }

  if (testStatus === 'completed') {
    return (
      <div className="max-w-lg mx-auto p-6 bg-white rounded-lg shadow-lg text-center">
        <CheckCircle className="w-16 h-16 text-green-600 mx-auto mb-4" />
        <h2 className="text-2xl font-bold text-gray-900 mb-4">Test Completed!</h2>
        <p className="text-gray-600 mb-4">
          Thank you for completing the neuropathy test. Your results have been recorded.
        </p>
        <div className="text-sm text-gray-500">
          <p>Test duration: {formatTime(timeElapsed)}</p>
          <p>Stimuli completed: {testData?.completedStimuli}/{testData?.totalStimuli}</p>
        </div>
      </div>
    )
  }

  return (
    <div className="max-w-4xl mx-auto p-6 bg-white rounded-lg shadow-lg">
      {/* Header with progress */}
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-gray-900">Neuropathy Test</h2>
        <div className="flex items-center space-x-4 text-sm text-gray-600">
          <div className="flex items-center">
            <Clock className="w-4 h-4 mr-1" />
            {formatTime(timeElapsed)}
          </div>
          <div className="flex items-center">
            <Activity className="w-4 h-4 mr-1" />
            {testData?.completedStimuli}/{testData?.totalStimuli}
          </div>
        </div>
      </div>

      {/* Progress bar */}
      <div className="mb-6">
        <div className="flex justify-between text-sm text-gray-600 mb-1">
          <span>Progress</span>
          <span>{Math.round(testData?.progress || 0)}%</span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div 
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${testData?.progress || 0}%` }}
          ></div>
        </div>
      </div>

      {/* Real-time Risk Assessment */}
      {testStatus === 'in_progress' && testData?.completedStimuli > 0 && (
        <RealTimeRiskAssessment 
          testId={testId}
          completedResponses={testData.completedStimuli}
          onRiskUpdate={(riskData) => setCurrentRiskLevel(riskData.riskLevel)}
        />
      )}

      {currentStimulus && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Response Form */}
          <div className="space-y-6">
            <div className="text-center p-6 bg-gradient-to-br from-blue-50 to-indigo-100 rounded-lg border-2 border-blue-200 relative">
              {/* Stimulus presentation indicator */}
              {!currentStimulus.isControlTrial && awaitingResponse && (
                <div className="absolute top-2 right-2">
                  <div className="w-3 h-3 bg-green-400 rounded-full animate-pulse"></div>
                </div>
              )}
              
              <h3 className="text-xl font-medium text-blue-900 mb-3">
                Stimulus #{currentStimulus.sequence}
                {currentStimulus.isControlTrial && (
                  <span className="text-sm text-gray-600 ml-2">(Control)</span>
                )}
              </h3>
              
              {/* Visual feedback for stimulus type */}
              {!currentStimulus.isControlTrial && awaitingResponse && (
                <div className="mb-4">
                  {getStimulusIcon(currentStimulus.type)}
                  <p className="text-sm text-gray-600 mt-2">
                    {getStimulusMessage(currentStimulus.type)}
                  </p>
                </div>
              )}
              
              <p className="text-blue-700 text-lg font-medium">
                {currentStimulus.isControlTrial 
                  ? "Focus and report any sensations you feel"
                  : "A stimulus may be presented - report what you feel"
                }
              </p>
              
              {awaitingResponse && (
                <div className="mt-3 text-sm text-gray-500">
                  Please respond when ready
                </div>
              )}
            </div>

            {/* Felt Sensation */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Did you feel a sensation?
              </label>
              <div className="flex space-x-4">
                <button
                  onClick={() => setResponse(prev => ({ ...prev, feltSensation: true }))}
                  className={`flex-1 py-2 px-4 rounded-lg border-2 transition-colors ${
                    response.feltSensation
                      ? 'border-green-500 bg-green-50 text-green-700'
                      : 'border-gray-300 bg-white text-gray-700 hover:border-gray-400'
                  }`}
                >
                  Yes
                </button>
                <button
                  onClick={() => setResponse(prev => ({ 
                    ...prev, 
                    feltSensation: false,
                    perceivedType: '',
                    perceivedLocation: null
                  }))}
                  className={`flex-1 py-2 px-4 rounded-lg border-2 transition-colors ${
                    !response.feltSensation
                      ? 'border-red-500 bg-red-50 text-red-700'
                      : 'border-gray-300 bg-white text-gray-700 hover:border-gray-400'
                  }`}
                >
                  No
                </button>
              </div>
            </div>

            {response.feltSensation && (
              <>
                {/* Intensity */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Intensity (1 = very weak, 10 = very strong)
                  </label>
                  <div className="flex items-center space-x-4">
                    <span className="text-sm text-gray-500">1</span>
                    <input
                      type="range"
                      min="1"
                      max="10"
                      value={response.perceivedIntensity}
                      onChange={(e) => setResponse(prev => ({ 
                        ...prev, 
                        perceivedIntensity: parseInt(e.target.value) 
                      }))}
                      className="flex-1"
                    />
                    <span className="text-sm text-gray-500">10</span>
                    <span className="font-medium text-blue-600 w-8 text-center">
                      {response.perceivedIntensity}
                    </span>
                  </div>
                </div>

                {/* Type */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    What type of sensation did you feel?
                  </label>
                  <select
                    value={response.perceivedType}
                    onChange={(e) => setResponse(prev => ({ 
                      ...prev, 
                      perceivedType: e.target.value 
                    }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
                  >
                    <option value="">Select sensation type...</option>
                    {getStimulusTypeOptions().map(option => (
                      <option key={option.value} value={option.value}>
                        {option.label}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Confidence */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    How confident are you in your response? (1 = not sure, 5 = very sure)
                  </label>
                  <div className="flex items-center space-x-4">
                    <span className="text-sm text-gray-500">1</span>
                    <input
                      type="range"
                      min="1"
                      max="5"
                      value={response.responseConfidence}
                      onChange={(e) => setResponse(prev => ({ 
                        ...prev, 
                        responseConfidence: parseInt(e.target.value) 
                      }))}
                      className="flex-1"
                    />
                    <span className="text-sm text-gray-500">5</span>
                    <span className="font-medium text-blue-600 w-8 text-center">
                      {response.responseConfidence}
                    </span>
                  </div>
                </div>
              </>
            )}

            <button
              onClick={submitResponse}
              disabled={!awaitingResponse || (response.feltSensation && (!response.perceivedType || !response.perceivedLocation))}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-300 text-white py-3 px-4 rounded-lg font-medium transition-colors"
            >
              {awaitingResponse ? 'Submit Response' : 'Processing...'}
            </button>
          </div>

          {/* Foot Diagram */}
          {response.feltSensation && (
            <div className="space-y-4">
              <div className="flex justify-center space-x-4 mb-4">
                <button
                  onClick={() => setFootView('top')}
                  className={`px-4 py-2 rounded-lg transition-colors ${
                    footView === 'top'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  Top View
                </button>
                <button
                  onClick={() => setFootView('bottom')}
                  className={`px-4 py-2 rounded-lg transition-colors ${
                    footView === 'bottom'
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  Bottom View
                </button>
              </div>
              
              <FootDiagram
                onLocationSelect={(location) => setResponse(prev => ({ 
                  ...prev, 
                  perceivedLocation: location 
                }))}
                selectedLocation={response.perceivedLocation}
                viewType={footView}
              />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default PatientTestInterface