import { motion } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { Shield, FileText, CheckCircle, Lock } from 'lucide-react'
import Card from '@components/common/Card'

function Compliance() {
  return (
    <div className="space-y-6">
      <Helmet>
        <title>Compliance - Smart Shoe Monitor</title>
        <meta name="description" content="HIPAA compliance and data protection information" />
      </Helmet>

      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Compliance & Privacy</h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">Your data protection and regulatory compliance</p>
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card title="HIPAA Compliance">
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <CheckCircle className="w-5 h-5 text-green-600" />
              <span className="text-sm text-gray-700 dark:text-gray-300">Data encryption at rest and in transit</span>
            </div>
            <div className="flex items-center gap-3">
              <CheckCircle className="w-5 h-5 text-green-600" />
              <span className="text-sm text-gray-700 dark:text-gray-300">Access controls and audit logging</span>
            </div>
            <div className="flex items-center gap-3">
              <CheckCircle className="w-5 h-5 text-green-600" />
              <span className="text-sm text-gray-700 dark:text-gray-300">Regular security assessments</span>
            </div>
          </div>
        </Card>

        <Card title="Data Protection">
          <div className="text-center py-8">
            <Lock className="w-12 h-12 text-blue-600 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">Your Data is Secure</h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">We follow industry best practices to protect your medical information</p>
          </div>
        </Card>
      </motion.div>
    </div>
  )
}

export default Compliance