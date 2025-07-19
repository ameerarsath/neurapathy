import { motion } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { FileText, Users, TrendingUp, Award } from 'lucide-react'
import Card from '@components/common/Card'
import Button from '@components/common/Button'

function Research() {
  return (
    <div className="space-y-6">
      <Helmet>
        <title>Research - Smart Shoe Monitor</title>
        <meta name="description" content="Clinical research and study participation" />
      </Helmet>

      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Research & Studies</h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">Participate in clinical research to advance diabetic neuropathy treatment</p>
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card title="Available Studies">
          <div className="space-y-4">
            <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
              <h3 className="font-medium text-gray-900 dark:text-white mb-2">Neuropathy Progression Study</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">Long-term study tracking progression patterns</p>
              <Button size="sm">Learn More</Button>
            </div>
            <div className="p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
              <h3 className="font-medium text-gray-900 dark:text-white mb-2">ML Algorithm Validation</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">Help improve AI prediction accuracy</p>
              <Button size="sm">Learn More</Button>
            </div>
          </div>
        </Card>

        <Card title="Your Contributions">
          <div className="text-center py-8">
            <Award className="w-12 h-12 text-yellow-500 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">Research Contributor</h3>
            <p className="text-gray-600 dark:text-gray-400">Thank you for contributing to medical research!</p>
          </div>
        </Card>
      </motion.div>
    </div>
  )
}

export default Research