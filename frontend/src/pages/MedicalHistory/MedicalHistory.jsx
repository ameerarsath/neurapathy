import { useState } from 'react'
import { motion } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { FileText, Calendar, Plus, Filter } from 'lucide-react'
import Card from '@components/common/Card'
import Button from '@components/common/Button'

function MedicalHistory() {
  const [filter, setFilter] = useState('all')

  const mockHistory = [
    { id: 1, date: '2024-01-15', type: 'Test Result', title: 'Vibration Assessment', status: 'completed' },
    { id: 2, date: '2024-01-10', type: 'Medication', title: 'Metformin 500mg prescribed', status: 'active' },
    { id: 3, date: '2024-01-05', type: 'Appointment', title: 'Endocrinologist Consultation', status: 'completed' },
  ]

  return (
    <div className="space-y-6">
      <Helmet>
        <title>Medical History - Smart Shoe Monitor</title>
        <meta name="description" content="View your complete medical history and records" />
      </Helmet>

      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Medical History</h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">Your complete medical record and health timeline</p>
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex justify-between items-center">
        <div className="flex gap-2">
          {['all', 'tests', 'medications', 'appointments'].map((tab) => (
            <button key={tab} onClick={() => setFilter(tab)} className={`px-4 py-2 text-sm font-medium rounded-md transition-colors ${filter === tab ? 'bg-blue-100 text-blue-700' : 'text-gray-600 hover:text-gray-900'}`}>
              {tab.charAt(0).toUpperCase() + tab.slice(1)}
            </button>
          ))}
        </div>
        <Button className="flex items-center gap-2">
          <Plus className="w-4 h-4" />
          Add Entry
        </Button>
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
        <Card title="Timeline">
          <div className="space-y-4">
            {mockHistory.map((item, index) => (
              <div key={item.id} className="flex items-start gap-4 p-4 border border-gray-200 dark:border-gray-700 rounded-lg">
                <div className="w-10 h-10 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                  <FileText className="w-5 h-5 text-blue-600" />
                </div>
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <h3 className="font-medium text-gray-900 dark:text-white">{item.title}</h3>
                    <span className="text-sm text-gray-500">{item.date}</span>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400">{item.type}</p>
                  <span className={`inline-block px-2 py-1 text-xs rounded-full mt-2 ${item.status === 'completed' ? 'bg-green-100 text-green-800' : 'bg-blue-100 text-blue-800'}`}>
                    {item.status}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </Card>
      </motion.div>
    </div>
  )
}

export default MedicalHistory