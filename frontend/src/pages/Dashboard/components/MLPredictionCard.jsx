import { useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Brain, 
  TrendingUp, 
  TrendingDown, 
  AlertTriangle, 
  Info,
  RefreshCw,
  Eye,
  Calendar
} from 'lucide-react'
import Card from '@components/common/Card'
import Button from '@components/common/Button'
import LoadingSpinner from '@components/common/LoadingSpinner'

function MLPredictionCard({ predictions, loading }) {
  const [showDetails, setShowDetails] = useState(false)

  if (loading) {
    return (
      <Card title="ML Predictions" className="h-full">
        <div className="flex items-center justify-center h-40">
          <LoadingSpinner size="md" text="Loading predictions..." />
        </div>
      </Card>
    )
  }

  if (!predictions) {
    return (
      <Card title="ML Predictions" className="h-full">
        <div className="flex flex-col items-center justify-center h-40 text-gray-500 dark:text-gray-400">
          <Brain className="w-8 h-8 mb-2" />
          <p className="text-sm">No predictions available</p>
          <Button variant="outline" size="sm" className="mt-2">
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </Button>
        </div>
      </Card>
    )
  }

  const getRiskColor = (risk) => {
    switch (risk?.toLowerCase()) {
      case 'low': return 'text-green-600 bg-green-100 dark:bg-green-900/20'
      case 'medium': return 'text-yellow-600 bg-yellow-100 dark:bg-yellow-900/20'
      case 'high': return 'text-red-600 bg-red-100 dark:bg-red-900/20'
      default: return 'text-gray-600 bg-gray-100 dark:bg-gray-900/20'
    }
  }

  const getProgressionIcon = (trend) => {
    switch (trend?.toLowerCase()) {
      case 'improving': return <TrendingDown className="w-4 h-4 text-green-600" />
      case 'worsening': return <TrendingUp className="w-4 h-4 text-red-600" />
      case 'stable': return <AlertTriangle className="w-4 h-4 text-yellow-600" />
      default: return <Info className="w-4 h-4 text-gray-600" />
    }
  }

  return (
    <Card 
      title="ML Predictions" 
      className="h-full"
      actions={
        <Button
          variant="ghost"
          size="sm"
          onClick={() => setShowDetails(!showDetails)}
        >
          <Eye className="w-4 h-4" />
        </Button>
      }
    >
      <div className="space-y-4">
        {/* Risk Level */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-blue-600" />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Risk Level
            </span>
          </div>
          <span className={`px-2 py-1 text-xs font-semibold rounded-full ${getRiskColor(predictions.risk_level)}`}>
            {predictions.risk_level || 'Unknown'}
          </span>
        </div>

        {/* Progression Trend */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {getProgressionIcon(predictions.progression_trend)}
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Progression
            </span>
          </div>
          <span className="text-sm text-gray-600 dark:text-gray-400">
            {predictions.progression_trend || 'Stable'}
          </span>
        </div>

        {/* Confidence Score */}
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Confidence
            </span>
            <span className="text-sm text-gray-600 dark:text-gray-400">
              {Math.round((predictions.confidence || 0) * 100)}%
            </span>
          </div>
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
            <motion.div
              initial={{ width: 0 }}
              animate={{ width: `${(predictions.confidence || 0) * 100}%` }}
              transition={{ duration: 1, ease: "easeOut" }}
              className="bg-blue-600 h-2 rounded-full"
            />
          </div>
        </div>

        {/* Key Factors */}
        {predictions.key_factors && (
          <div className="space-y-2">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Key Risk Factors
            </span>
            <div className="space-y-1">
              {predictions.key_factors.slice(0, 3).map((factor, index) => (
                <div key={index} className="flex items-center justify-between text-xs">
                  <span className="text-gray-600 dark:text-gray-400">{factor.factor}</span>
                  <span className="font-medium">{factor.weight}%</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Next Assessment */}
        <div className="pt-3 border-t border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
            <Calendar className="w-3 h-3" />
            <span>Next assessment: {predictions.next_assessment || '7 days'}</span>
          </div>
        </div>

        {/* Detailed View */}
        {showDetails && predictions.detailed_analysis && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="pt-3 border-t border-gray-200 dark:border-gray-700"
          >
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Detailed Analysis
              </h4>
              <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                {predictions.detailed_analysis.split('\n').map((line, index) => (
                  <p key={index}>{line}</p>
                ))}
              </div>
            </div>
          </motion.div>
        )}

        {/* Recommendations */}
        {predictions.recommendations && (
          <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
            <h4 className="text-sm font-medium text-blue-800 dark:text-blue-200 mb-2">
              Recommendations
            </h4>
            <ul className="text-xs text-blue-700 dark:text-blue-300 space-y-1">
              {predictions.recommendations.slice(0, 2).map((rec, index) => (
                <li key={index} className="flex items-start gap-1">
                  <span className="text-blue-500 mt-0.5">•</span>
                  <span>{rec}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Action Button */}
        <Button
          variant="outline"
          size="sm"
          className="w-full mt-4"
          onClick={() => window.location.href = '/ml-predictions'}
        >
          View Full Analysis
        </Button>
      </div>
    </Card>
  )
}

export default MLPredictionCard