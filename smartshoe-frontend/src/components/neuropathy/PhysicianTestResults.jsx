import React, { useState, useEffect } from 'react'
import { CheckCircle, XCircle, AlertTriangle, TrendingUp, TrendingDown, Clock, Target, Brain, Activity, FileText } from 'lucide-react'
import api from '../../services/api'
import toast from 'react-hot-toast'

const PhysicianTestResults = ({ testId }) => {
  const [testResults, setTestResults] = useState(null)
  const [loading, setLoading] = useState(true)
  const [selectedStimulus, setSelectedStimulus] = useState(null)
  const [filterType, setFilterType] = useState('all')
  const [mlAnalysis, setMlAnalysis] = useState(null)
  const [mlLoading, setMlLoading] = useState(false)
  const [showMLAnalysis, setShowMLAnalysis] = useState(false)

  useEffect(() => {
    loadTestResults()
  }, [testId])

  const loadTestResults = async () => {
    try {
      setLoading(true)
      const response = await api.get(`/api/neuropathy/test/${testId}/physician-results`)
      if (response.data.success) {
        setTestResults(response.data)
        
        // If ML analysis is included in the response, set it
        if (response.data.mlAnalysis) {
          setMlAnalysis(response.data.mlAnalysis)
          setShowMLAnalysis(true)
        }
      }
    } catch (error) {
      console.error('Error loading test results:', error)
      toast.error('Failed to load test results')
    } finally {
      setLoading(false)
    }
  }
  
  const loadMLAnalysis = async () => {
    try {
      setMlLoading(true)
      const response = await api.get(`/api/neuropathy/test/${testId}/ml-analysis`)
      if (response.data.success) {
        setMlAnalysis(response.data.mlAnalysis)
        setShowMLAnalysis(true)
        toast.success('ML analysis completed')
      }
    } catch (error) {
      console.error('Error loading ML analysis:', error)
      toast.error('Failed to load ML analysis')
    } finally {
      setMlLoading(false)
    }
  }

  const getAccuracyColor = (accuracy) => {
    if (accuracy >= 0.8) return 'text-green-600'
    if (accuracy >= 0.6) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getAccuracyIcon = (accuracy) => {
    if (accuracy >= 0.8) return <CheckCircle className="w-4 h-4" />
    if (accuracy >= 0.6) return <AlertTriangle className="w-4 h-4" />
    return <XCircle className="w-4 h-4" />
  }

  const getStimulusTypeColor = (type) => {
    const colors = {
      'VIBRATION': 'bg-blue-100 text-blue-800',
      'TEMPERATURE_HOT': 'bg-red-100 text-red-800',
      'TEMPERATURE_COLD': 'bg-cyan-100 text-cyan-800',
      'PINPRICK': 'bg-purple-100 text-purple-800',
      'PRESSURE': 'bg-green-100 text-green-800',
      'NONE': 'bg-gray-100 text-gray-800'
    }
    return colors[type] || 'bg-gray-100 text-gray-800'
  }

  const formatDuration = (startTime, endTime) => {
    if (!startTime || !endTime) return 'N/A'
    const start = new Date(startTime)
    const end = new Date(endTime)
    const diffMs = end - start
    const minutes = Math.floor(diffMs / 60000)
    const seconds = Math.floor((diffMs % 60000) / 1000)
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }

  const getFilteredStimuli = () => {
    if (!testResults?.stimuli) return []
    
    switch (filterType) {
      case 'correct':
        return testResults.stimuli.filter(s => s.correctDetection)
      case 'incorrect':
        return testResults.stimuli.filter(s => !s.correctDetection)
      case 'no-stimulus':
        return testResults.stimuli.filter(s => s.noStimulusTrial)
      default:
        return testResults.stimuli
    }
  }

  const calculateSeverityAssessment = () => {
    if (!testResults?.analytics) return null
    
    const accuracy = testResults.analytics.accuracy
    const totalStimuli = testResults.analytics.totalStimuli
    
    if (accuracy >= 0.9) return { level: 'Normal', color: 'text-green-600', description: 'Excellent sensation detection' }
    if (accuracy >= 0.7) return { level: 'Mild Impairment', color: 'text-yellow-600', description: 'Some sensory loss detected' }
    if (accuracy >= 0.5) return { level: 'Moderate Impairment', color: 'text-orange-600', description: 'Significant sensory deficits' }
    return { level: 'Severe Impairment', color: 'text-red-600', description: 'Major sensory loss, requires immediate attention' }
  }

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'SEVERE': return 'text-red-600 bg-red-50 border-red-200'
      case 'MODERATE': return 'text-orange-600 bg-orange-50 border-orange-200'
      case 'MILD': return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'NORMAL': return 'text-green-600 bg-green-50 border-green-200'
      default: return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'HIGH': return 'text-red-600 bg-red-50'
      case 'MODERATE': return 'text-orange-600 bg-orange-50'
      case 'LOW': return 'text-yellow-600 bg-yellow-50'
      case 'MINIMAL': return 'text-green-600 bg-green-50'
      default: return 'text-gray-600 bg-gray-50'
    }
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <span className="ml-2 text-gray-600">Loading test results...</span>
      </div>
    )
  }

  if (!testResults) {
    return (
      <div className="text-center p-8">
        <p className="text-gray-600">No test results available</p>
      </div>
    )
  }

  const severity = calculateSeverityAssessment()

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Test Summary */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Neuropathy Test Results</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-6">
          <div className="bg-blue-50 p-4 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-blue-600 font-medium">Overall Accuracy</p>
                <p className="text-2xl font-bold text-blue-900">
                  {Math.round(testResults.analytics.accuracy * 100)}%
                </p>
              </div>
              <Target className="w-8 h-8 text-blue-600" />
            </div>
          </div>
          
          <div className="bg-green-50 p-4 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-green-600 font-medium">Correct Detections</p>
                <p className="text-2xl font-bold text-green-900">
                  {testResults.analytics.correctDetections}/{testResults.analytics.totalStimuli}
                </p>
              </div>
              <CheckCircle className="w-8 h-8 text-green-600" />
            </div>
          </div>
          
          <div className="bg-purple-50 p-4 rounded-lg">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-purple-600 font-medium">Test Duration</p>
                <p className="text-2xl font-bold text-purple-900">
                  {formatDuration(testResults.test.startedAt, testResults.test.completedAt)}
                </p>
              </div>
              <Clock className="w-8 h-8 text-purple-600" />
            </div>
          </div>
          
          <div className={`p-4 rounded-lg ${severity?.level === 'Normal' ? 'bg-green-50' : severity?.level.includes('Mild') ? 'bg-yellow-50' : 'bg-red-50'}`}>
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium" style={{ color: severity?.color.replace('text-', '') }}>
                  Assessment
                </p>
                <p className={`text-lg font-bold ${severity?.color}`}>
                  {severity?.level}
                </p>
              </div>
              {severity?.level === 'Normal' ? 
                <TrendingUp className="w-8 h-8 text-green-600" /> :
                <TrendingDown className="w-8 h-8 text-red-600" />
              }
            </div>
          </div>
        </div>
        
        {severity && (
          <div className={`p-4 rounded-lg border-l-4 ${
            severity.level === 'Normal' ? 'bg-green-50 border-green-400' :
            severity.level.includes('Mild') ? 'bg-yellow-50 border-yellow-400' :
            'bg-red-50 border-red-400'
          }`}>
            <h4 className={`font-medium ${severity.color}`}>Clinical Assessment</h4>
            <p className="text-sm text-gray-700 mt-1">{severity.description}</p>
          </div>
        )}
      </div>

      {/* ML Analysis Section */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-xl font-bold text-gray-900 flex items-center">
            <Brain className="w-6 h-6 mr-2 text-purple-600" />
            AI-Powered Clinical Analysis
          </h3>
          
          {!showMLAnalysis && testResults?.test?.status === 'COMPLETED' && (
            <button
              onClick={loadMLAnalysis}
              disabled={mlLoading}
              className="bg-purple-600 hover:bg-purple-700 disabled:bg-gray-300 text-white px-4 py-2 rounded-lg font-medium flex items-center space-x-2"
            >
              {mlLoading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Activity className="w-4 h-4" />
                  <span>Run ML Analysis</span>
                </>
              )}
            </button>
          )}
        </div>

        {showMLAnalysis && mlAnalysis ? (
          <div className="space-y-6">
            {/* Severity Assessment */}
            <div className={`p-6 rounded-lg border-2 ${getSeverityColor(mlAnalysis.severity)}`}>
              <div className="flex items-center justify-between mb-4">
                <h4 className="text-lg font-bold">Neuropathy Severity Assessment</h4>
                <div className={`px-3 py-1 rounded-full text-sm font-medium ${getSeverityColor(mlAnalysis.severity)}`}>
                  {mlAnalysis.severity}
                </div>
              </div>
              <p className="text-sm mb-4">{mlAnalysis.severityInterpretation}</p>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className={`p-4 rounded-lg ${getRiskColor(mlAnalysis.riskLevel)}`}>
                  <div className="text-sm font-medium mb-1">Risk Score</div>
                  <div className="text-2xl font-bold">{Math.round(mlAnalysis.riskScore * 100)}%</div>
                  <div className="text-xs">{mlAnalysis.riskLevel} Risk</div>
                </div>
                
                <div className="p-4 bg-blue-50 rounded-lg">
                  <div className="text-sm font-medium text-blue-600 mb-1">ML Confidence</div>
                  <div className="text-2xl font-bold text-blue-900">{Math.round(mlAnalysis.confidenceScore * 100)}%</div>
                  <div className="text-xs text-blue-600">Analysis Reliability</div>
                </div>
                
                <div className="p-4 bg-gray-50 rounded-lg">
                  <div className="text-sm font-medium text-gray-600 mb-1">Detailed Metrics</div>
                  <div className="text-xs text-gray-600">
                    {Object.keys(mlAnalysis.detailedAnalysis).length} parameters analyzed
                  </div>
                  <button 
                    onClick={() => setShowMLAnalysis(prev => !prev)}
                    className="text-blue-600 hover:text-blue-800 text-xs mt-1"
                  >
                    View Details →
                  </button>
                </div>
              </div>
            </div>

            {/* Detailed Analysis Scores */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              {Object.entries(mlAnalysis.detailedAnalysis).map(([key, value]) => (
                <div key={key} className="bg-gray-50 p-4 rounded-lg">
                  <div className="text-xs text-gray-600 mb-1">
                    {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </div>
                  <div className={`text-lg font-bold ${getAccuracyColor(value)}`}>
                    {Math.round(value * 100)}%
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-1.5 mt-2">
                    <div 
                      className={`h-1.5 rounded-full ${
                        value >= 0.8 ? 'bg-green-500' : 
                        value >= 0.6 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${value * 100}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>

            {/* Clinical Recommendations */}
            <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
              <h5 className="font-bold text-blue-900 mb-4 flex items-center">
                <FileText className="w-5 h-5 mr-2" />
                AI-Generated Clinical Recommendations
              </h5>
              <ul className="space-y-2">
                {mlAnalysis.recommendations.map((recommendation, index) => (
                  <li key={index} className="flex items-start space-x-2">
                    <div className="w-2 h-2 bg-blue-600 rounded-full mt-2 flex-shrink-0"></div>
                    <span className="text-blue-800 text-sm">{recommendation}</span>
                  </li>
                ))}
              </ul>
            </div>
          </div>
        ) : !showMLAnalysis && testResults?.test?.status !== 'COMPLETED' ? (
          <div className="text-center py-8 text-gray-500">
            <Brain className="w-12 h-12 mx-auto mb-4 text-gray-400" />
            <p>ML analysis will be available after test completion</p>
          </div>
        ) : null}
      </div>

      {/* Detailed Results */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-xl font-bold text-gray-900">Stimulus-Response Analysis</h3>
          
          <div className="flex space-x-2">
            <select
              value={filterType}
              onChange={(e) => setFilterType(e.target.value)}
              className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Stimuli</option>
              <option value="correct">Correct Responses</option>
              <option value="incorrect">Incorrect Responses</option>
              <option value="no-stimulus">Control Trials</option>
            </select>
          </div>
        </div>
        
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Sequence
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actual Stimulus
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Patient Response
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Intensity
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Accuracy
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Details
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {getFilteredStimuli().map((stimulus) => (
                <tr 
                  key={stimulus.id}
                  className={`hover:bg-gray-50 cursor-pointer ${
                    selectedStimulus?.id === stimulus.id ? 'bg-blue-50' : ''
                  }`}
                  onClick={() => setSelectedStimulus(stimulus)}
                >
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                    #{stimulus.sequence}
                  </td>
                  
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex flex-col">
                      <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded-full ${getStimulusTypeColor(stimulus.actualStimulusType)}`}>
                        {stimulus.actualStimulusType.replace('_', ' ')}
                      </span>
                      {stimulus.noStimulusTrial && (
                        <span className="text-xs text-gray-500 mt-1">Control Trial</span>
                      )}
                    </div>
                  </td>
                  
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex flex-col">
                      <span className={`text-sm ${stimulus.patientFeltSensation ? 'text-green-600' : 'text-red-600'}`}>
                        {stimulus.patientFeltSensation ? 'Felt Sensation' : 'No Sensation'}
                      </span>
                      {stimulus.perceivedType && (
                        <span className="text-xs text-gray-500">
                          {stimulus.perceivedType.replace('_', ' ')}
                        </span>
                      )}
                    </div>
                  </td>
                  
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                    <div className="flex flex-col">
                      <span>Actual: {stimulus.actualIntensity ? (stimulus.actualIntensity * 10).toFixed(1) : '0'}</span>
                      <span>Perceived: {stimulus.perceivedIntensity || 'N/A'}</span>
                    </div>
                  </td>
                  
                  <td className="px-6 py-4 whitespace-nowrap">
                    <div className="flex items-center space-x-2">
                      <span className={getAccuracyColor(stimulus.intensityAccuracy)}>
                        {getAccuracyIcon(stimulus.intensityAccuracy)}
                      </span>
                      <span className={`text-sm font-medium ${getAccuracyColor(stimulus.intensityAccuracy)}`}>
                        {Math.round(stimulus.intensityAccuracy * 100)}%
                      </span>
                    </div>
                  </td>
                  
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        setSelectedStimulus(stimulus)
                      }}
                      className="text-blue-600 hover:text-blue-900"
                    >
                      View Details
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Detailed Stimulus View */}
      {selectedStimulus && (
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h4 className="text-lg font-bold text-gray-900 mb-4">
            Stimulus #{selectedStimulus.sequence} - Detailed Analysis
          </h4>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h5 className="font-medium text-gray-900 mb-3">Device Stimulus</h5>
              <div className="space-y-2 text-sm">
                <p><span className="font-medium">Type:</span> {selectedStimulus.actualStimulusType.replace('_', ' ')}</p>
                <p><span className="font-medium">Intensity:</span> {selectedStimulus.actualIntensity ? (selectedStimulus.actualIntensity * 10).toFixed(1) : '0'}/10</p>
                <p><span className="font-medium">Duration:</span> {selectedStimulus.duration}ms</p>
                <p><span className="font-medium">Location:</span> {selectedStimulus.actualLocation}</p>
                <p><span className="font-medium">Control Trial:</span> {selectedStimulus.noStimulusTrial ? 'Yes' : 'No'}</p>
              </div>
            </div>
            
            <div>
              <h5 className="font-medium text-gray-900 mb-3">Patient Response</h5>
              <div className="space-y-2 text-sm">
                <p><span className="font-medium">Felt Sensation:</span> 
                  <span className={selectedStimulus.patientFeltSensation ? 'text-green-600' : 'text-red-600'}>
                    {selectedStimulus.patientFeltSensation ? ' Yes' : ' No'}
                  </span>
                </p>
                {selectedStimulus.patientFeltSensation && (
                  <>
                    <p><span className="font-medium">Perceived Type:</span> {selectedStimulus.perceivedType?.replace('_', ' ') || 'Not specified'}</p>
                    <p><span className="font-medium">Perceived Intensity:</span> {selectedStimulus.perceivedIntensity || 'N/A'}/10</p>
                    <p><span className="font-medium">Confidence:</span> {selectedStimulus.responseConfidence || 'N/A'}/5</p>
                    <p><span className="font-medium">Response Location:</span> {selectedStimulus.perceivedLocation || 'Not specified'}</p>
                  </>
                )}
              </div>
            </div>
          </div>
          
          <div className="mt-6 p-4 bg-gray-50 rounded-lg">
            <h6 className="font-medium text-gray-900 mb-2">Clinical Interpretation</h6>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div>
                <span className="font-medium">Detection Accuracy:</span>
                <span className={`ml-2 ${selectedStimulus.correctDetection ? 'text-green-600' : 'text-red-600'}`}>
                  {selectedStimulus.correctDetection ? 'Correct' : 'Incorrect'}
                </span>
              </div>
              <div>
                <span className="font-medium">Type Recognition:</span>
                <span className={`ml-2 ${selectedStimulus.typeAccuracy ? 'text-green-600' : 'text-red-600'}`}>
                  {selectedStimulus.typeAccuracy ? 'Correct' : 'Incorrect'}
                </span>
              </div>
              <div>
                <span className="font-medium">Intensity Accuracy:</span>
                <span className={`ml-2 ${getAccuracyColor(selectedStimulus.intensityAccuracy)}`}>
                  {Math.round(selectedStimulus.intensityAccuracy * 100)}%
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default PhysicianTestResults