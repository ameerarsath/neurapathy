import { useState } from 'react'
import { motion } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { 
  FileText, 
  Download, 
  Calendar,
  Filter,
  BarChart3,
  TrendingUp,
  Users,
  Activity
} from 'lucide-react'

import Card from '@components/common/Card'
import Button from '@components/common/Button'

function Reports() {
  const [selectedReport, setSelectedReport] = useState('patient-summary')
  const [dateRange, setDateRange] = useState('month')

  const reportTypes = [
    { id: 'patient-summary', name: 'Patient Summary', icon: Users },
    { id: 'progression-analysis', name: 'Progression Analysis', icon: TrendingUp },
    { id: 'device-usage', name: 'Device Usage', icon: Activity },
    { id: 'test-compliance', name: 'Test Compliance', icon: BarChart3 },
  ]

  return (
    <div className="space-y-6">
      <Helmet>
        <title>Reports - Smart Shoe Monitor</title>
        <meta name="description" content="Generate and download medical reports" />
      </Helmet>

      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col lg:flex-row lg:items-center lg:justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Reports
          </h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            Generate comprehensive medical reports and analytics
          </p>
        </div>
        
        <Button className="mt-4 lg:mt-0 flex items-center gap-2">
          <Download className="w-4 h-4" />
          Generate Report
        </Button>
      </motion.div>

      {/* Report Types */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4"
      >
        {reportTypes.map((report, index) => {
          const Icon = report.icon
          return (
            <motion.div
              key={report.id}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: index * 0.1 }}
              onClick={() => setSelectedReport(report.id)}
              className={`cursor-pointer transition-all ${
                selectedReport === report.id
                  ? 'ring-2 ring-blue-500'
                  : 'hover:shadow-md'
              }`}
            >
              <Card className="p-6 text-center">
                <Icon className="w-8 h-8 text-blue-600 mx-auto mb-3" />
                <h3 className="font-medium text-gray-900 dark:text-white">
                  {report.name}
                </h3>
              </Card>
            </motion.div>
          )
        })}
      </motion.div>

      {/* Report Configuration */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <Card title="Report Configuration">
          <div className="space-y-4">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Date Range
                </label>
                <select
                  value={dateRange}
                  onChange={(e) => setDateRange(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
                >
                  <option value="week">Last Week</option>
                  <option value="month">Last Month</option>
                  <option value="quarter">Last Quarter</option>
                  <option value="year">Last Year</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Format
                </label>
                <select className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white">
                  <option value="pdf">PDF</option>
                  <option value="csv">CSV</option>
                  <option value="excel">Excel</option>
                </select>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Include
                </label>
                <select className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white">
                  <option value="all">All Data</option>
                  <option value="summary">Summary Only</option>
                  <option value="detailed">Detailed Analysis</option>
                </select>
              </div>
            </div>
            
            <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
              <Button className="w-full md:w-auto">
                <Download className="w-4 h-4 mr-2" />
                Generate {reportTypes.find(r => r.id === selectedReport)?.name} Report
              </Button>
            </div>
          </div>
        </Card>
      </motion.div>

      {/* Recent Reports */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
      >
        <Card title="Recent Reports">
          <div className="space-y-3">
            {[1, 2, 3].map((item) => (
              <div key={item} className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
                <div className="flex items-center gap-3">
                  <FileText className="w-5 h-5 text-blue-600" />
                  <div>
                    <div className="font-medium text-gray-900 dark:text-white">
                      Patient Summary Report
                    </div>
                    <div className="text-sm text-gray-600 dark:text-gray-400">
                      Generated on Jan 15, 2024
                    </div>
                  </div>
                </div>
                <Button variant="outline" size="sm">
                  <Download className="w-4 h-4" />
                </Button>
              </div>
            ))}
          </div>
        </Card>
      </motion.div>
    </div>
  )
}

export default Reports