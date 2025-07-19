import { useState } from 'react'
import { useQuery } from 'react-query'
import { motion } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { 
  Users, 
  Plus, 
  Search, 
  User,
  Activity,
  Calendar,
  AlertTriangle,
  CheckCircle
} from 'lucide-react'

import { api } from '@services/api'
import Card from '@components/common/Card'
import Button from '@components/common/Button'
import Input from '@components/common/Input'
import LoadingSpinner from '@components/common/LoadingSpinner'

function PatientManagement() {
  const [searchTerm, setSearchTerm] = useState('')

  const { data: patients, isLoading } = useQuery(
    ['patients', searchTerm],
    () => api.patient.getPatients({ search: searchTerm }),
    { staleTime: 2 * 60 * 1000 }
  )

  if (isLoading) {
    return <div className="flex items-center justify-center min-h-screen">
      <LoadingSpinner size="lg" text="Loading patients..." />
    </div>
  }

  // Use API data or empty array if no data
  const displayPatients = patients || []

  return (
    <div className="space-y-6">
      <Helmet>
        <title>Patient Management - Smart Shoe Monitor</title>
        <meta name="description" content="Manage patient records and monitoring" />
      </Helmet>

      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Patient Management</h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">Monitor and manage patient health records</p>
        </div>
        <Button className="flex items-center gap-2">
          <Plus className="w-4 h-4" />
          Add Patient
        </Button>
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <Input type="text" placeholder="Search patients..." value={searchTerm} onChange={(e) => setSearchTerm(e.target.value)} icon={Search} />
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {displayPatients.map((patient, index) => (
          <Card key={patient.id} className="p-6 hover:shadow-lg transition-shadow cursor-pointer">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center">
                  <User className="w-5 h-5 text-blue-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white">{patient.name}</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Age: {patient.age}</p>
                </div>
              </div>
              <span className={`px-2 py-1 text-xs font-medium rounded-full ${
                patient.riskLevel === 'high' ? 'bg-red-100 text-red-800' :
                patient.riskLevel === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                'bg-green-100 text-green-800'
              }`}>
                {patient.riskLevel}
              </span>
            </div>
            <div className="space-y-2 text-sm">
              <div className="flex items-center gap-2">
                <Calendar className="w-4 h-4 text-gray-500" />
                <span className="text-gray-600 dark:text-gray-400">Last test: {patient.lastTest}</span>
              </div>
              <div className="flex items-center gap-2">
                {patient.status === 'active' ? <CheckCircle className="w-4 h-4 text-green-500" /> : <AlertTriangle className="w-4 h-4 text-red-500" />}
                <span className="text-gray-600 dark:text-gray-400">Status: {patient.status}</span>
              </div>
            </div>
          </Card>
        ))}
      </motion.div>
    </div>
  )
}

export default PatientManagement