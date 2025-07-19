import React, { useState, useEffect } from 'react'
import { AlertTriangle, TrendingUp, TrendingDown, Activity, Shield } from 'lucide-react'
import api from '../../services/api'

const RealTimeRiskAssessment = ({ testId, completedResponses, onRiskUpdate }) => {
  const [riskAssessment, setRiskAssessment] = useState(null)
  const [loading, setLoading] = useState(false)
  const [showDetails, setShowDetails] = useState(false)

  useEffect(() => {
    // Trigger risk assessment after every 5 completed responses
    if (completedResponses >= 5 && completedResponses % 5 === 0) {
      performRiskAssessment()
    }
  }, [completedResponses, testId])

  const performRiskAssessment = async () => {
    try {
      setLoading(true)
      const response = await api.post(`/api/neuropathy/test/${testId}/risk-assessment`)
      
      if (response.data.success) {
        setRiskAssessment(response.data.riskAssessment)
        
        // Notify parent component of risk level change
        if (onRiskUpdate) {
          onRiskUpdate(response.data.riskAssessment)
        }
      }
    } catch (error) {
      console.error('Error performing risk assessment:', error)
    } finally {
      setLoading(false)
    }
  }

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'HIGH': return 'text-red-600 bg-red-50 border-red-200'
      case 'MODERATE': return 'text-orange-600 bg-orange-50 border-orange-200'
      case 'LOW': return 'text-yellow-600 bg-yellow-50 border-yellow-200'
      case 'MINIMAL': return 'text-green-600 bg-green-50 border-green-200'
      default: return 'text-gray-600 bg-gray-50 border-gray-200'
    }
  }

  const getRiskIcon = (riskLevel) => {
    switch (riskLevel) {
      case 'HIGH': return <AlertTriangle className="w-5 h-5 text-red-600" />
      case 'MODERATE': return <TrendingDown className="w-5 h-5 text-orange-600" />
      case 'LOW': return <Activity className="w-5 h-5 text-yellow-600" />
      case 'MINIMAL': return <Shield className="w-5 h-5 text-green-600" />
      default: return <Activity className="w-5 h-5 text-gray-600" />
    }
  }

  const getRiskMessage = (riskLevel) => {
    switch (riskLevel) {
      case 'HIGH':
        return 'Elevated risk detected based on current responses. Consider clinical review.'
      case 'MODERATE':
        return 'Moderate risk indicators present. Continue monitoring closely.'
      case 'LOW':
        return 'Some risk factors identified. Regular monitoring recommended.'
      case 'MINIMAL':
        return 'Low risk profile based on current responses. Continue preventive care.'
      default:
        return 'Risk assessment in progress...'
    }
  }

  if (!riskAssessment && completedResponses < 5) {
    return null // Don't show until we have enough data
  }

  return (
    <div className="mb-4">
      {loading && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
            <span className="text-blue-700 font-medium">Analyzing risk factors...</span>
          </div>
        </div>
      )}

      {riskAssessment && (
        <div className={`border-2 rounded-lg p-4 ${getRiskColor(riskAssessment.riskLevel)}`}>
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center space-x-3">
              {getRiskIcon(riskAssessment.riskLevel)}
              <div>
                <h4 className="font-bold text-sm">Real-Time Risk Assessment</h4>
                <p className="text-xs opacity-75">
                  Based on {riskAssessment.samplesAnalyzed} responses
                </p>
              </div>
            </div>
            
            <div className="text-right">
              <div className="text-lg font-bold">
                {Math.round(riskAssessment.preliminaryRiskScore * 100)}%
              </div>
              <div className="text-xs font-medium">
                {riskAssessment.riskLevel} RISK
              </div>
            </div>
          </div>

          <p className="text-sm mb-3">
            {getRiskMessage(riskAssessment.riskLevel)}
          </p>

          {riskAssessment.earlyWarning && (
            <div className="bg-red-100 border border-red-300 rounded p-3 mb-3">
              <div className="flex items-center space-x-2">
                <AlertTriangle className="w-4 h-4 text-red-600" />
                <span className="text-red-800 font-medium text-sm">
                  {riskAssessment.earlyWarning}
                </span>
              </div>
            </div>
          )}

          <button
            onClick={() => setShowDetails(!showDetails)}
            className="text-xs hover:underline opacity-75"
          >
            {showDetails ? 'Hide Details' : 'Show Details'}
          </button>

          {showDetails && (
            <div className="mt-3 pt-3 border-t border-current border-opacity-20">
              <div className="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <span className="font-medium">Detection Accuracy:</span>
                  <div className="mt-1">
                    <div className="flex justify-between">
                      <span>{Math.round(riskAssessment.partialAccuracy * 100)}%</span>
                      <span className="opacity-75">Current</span>
                    </div>
                    <div className="w-full bg-white bg-opacity-30 rounded-full h-1.5 mt-1">
                      <div 
                        className="h-1.5 rounded-full bg-current opacity-75"
                        style={{ width: `${riskAssessment.partialAccuracy * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
                
                <div>
                  <span className="font-medium">False Positive Rate:</span>
                  <div className="mt-1">
                    <div className="flex justify-between">
                      <span>{Math.round(riskAssessment.falsePositiveRate * 100)}%</span>
                      <span className="opacity-75">Current</span>
                    </div>
                    <div className="w-full bg-white bg-opacity-30 rounded-full h-1.5 mt-1">
                      <div 
                        className="h-1.5 rounded-full bg-current opacity-75"
                        style={{ width: `${riskAssessment.falsePositiveRate * 100}%` }}
                      ></div>
                    </div>
                  </div>
                </div>
              </div>
              
              <div className="mt-3 text-xs opacity-75">
                <p>Note: This is a preliminary assessment based on partial data. 
                   Final analysis will be available after test completion.</p>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default RealTimeRiskAssessment