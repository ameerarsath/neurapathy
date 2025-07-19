import { useState } from 'react'
import { useQuery } from 'react-query'
import { motion } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { 
  BarChart3, 
  TrendingUp, 
  Activity, 
  Calendar,
  Download,
  Filter,
  Users,
  Target,
  AlertTriangle,
  CheckCircle
} from 'lucide-react'

import { useAuth } from '@contexts/AuthContext'
import { api } from '@services/api'
import Card from '@components/common/Card'
import Button from '@components/common/Button'
import LoadingSpinner from '@components/common/LoadingSpinner'

function Analytics() {
  const [timeRange, setTimeRange] = useState('month')
  const [analysisType, setAnalysisType] = useState('progression')
  
  const { user } = useAuth()

  const { data: analyticsData, isLoading } = useQuery(
    ['analytics', user?.id, timeRange, analysisType],
    () => api.analytics.getDetailedAnalytics(user?.id, { timeRange, analysisType }),
    {
      enabled: !!user?.id,
      staleTime: 5 * 60 * 1000,
    }
  )

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <LoadingSpinner size="lg" text="Loading analytics..." />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <Helmet>
        <title>Analytics - Smart Shoe Monitor</title>
        <meta name="description" content="Comprehensive health analytics and trend analysis" />
      </Helmet>

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col lg:flex-row lg:items-center lg:justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Health Analytics
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            Comprehensive analysis of your neuropathy progression and health trends
          </p>
        </div>
        
        <div className="mt-4 lg:mt-0 flex flex-wrap gap-2">
          <Button
            variant="outline"
            size="sm"
            className="flex items-center gap-2"
          >
            <Filter className="w-4 h-4" />
            Filter
          </Button>
          
          <Button
            variant="outline"
            size="sm"
            className="flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            Export Report
          </Button>
        </div>
      </motion.div>

      {/* Analytics Content */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="grid grid-cols-1 lg:grid-cols-3 gap-6"
      >
        {/* Key Metrics */}
        <Card title="Key Metrics" className="lg:col-span-1">
          <div className="space-y-4">
            <div className="flex items-center justify-between p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
              <div className="flex items-center gap-2">
                <TrendingUp className="w-4 h-4 text-blue-600" />
                <span className="text-sm font-medium">Progression Rate</span>
              </div>
              <span className="text-sm font-semibold text-blue-600">{analyticsData?.progressionRate || "N/A"}</span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-green-50 dark:bg-green-900/20 rounded-lg">
              <div className="flex items-center gap-2">
                <CheckCircle className="w-4 h-4 text-green-600" />
                <span className="text-sm font-medium">Test Compliance</span>
              </div>
              <span className="text-sm font-semibold text-green-600">{analyticsData?.testCompliance || "N/A"}</span>
            </div>
            
            <div className="flex items-center justify-between p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
              <div className="flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-yellow-600" />
                <span className="text-sm font-medium">Risk Level</span>
              </div>
              <span className="text-sm font-semibold text-yellow-600">{analyticsData?.riskLevel || "Unknown"}</span>
            </div>
          </div>
        </Card>

        {/* Chart Placeholder */}
        <Card title="Progression Trends" className="lg:col-span-2">
          <div className="h-64 flex items-center justify-center bg-gray-50 dark:bg-gray-800 rounded-lg">
            <div className="text-center">
              <BarChart3 className="w-12 h-12 text-gray-400 mx-auto mb-2" />
              <p className="text-gray-500">Chart visualization would go here</p>
            </div>
          </div>
        </Card>
      </motion.div>

      {/* Detailed Analysis */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <Card title="Detailed Analysis">
          <div className="space-y-4">
            <p className="text-gray-600 dark:text-gray-400">
              Your health analytics show a stable progression pattern with good test compliance. 
              Continue regular monitoring and follow medical recommendations.
            </p>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-2xl font-bold text-blue-600">{analyticsData?.totalTests || 0}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Total Tests</div>
              </div>
              <div className="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-2xl font-bold text-green-600">{analyticsData?.daysActive || 0}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Days Active</div>
              </div>
              <div className="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-2xl font-bold text-purple-600">{analyticsData?.deviceCount || 0}</div>
                <div className="text-sm text-gray-600 dark:text-gray-400">Devices</div>
              </div>
            </div>
          </div>
        </Card>
      </motion.div>
    </div>
  )
}

export default Analytics