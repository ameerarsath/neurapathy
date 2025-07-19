import { useState } from 'react'
import { motion } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { Pill, Plus, Clock, AlertTriangle } from 'lucide-react'
import Card from '@components/common/Card'
import Button from '@components/common/Button'

function Medications() {
  const mockMedications = [
    { id: 1, name: 'Metformin', dosage: '500mg', frequency: 'Twice daily', nextDose: '2:00 PM', status: 'active' },
    { id: 2, name: 'Lisinopril', dosage: '10mg', frequency: 'Once daily', nextDose: '8:00 AM', status: 'active' },
    { id: 3, name: 'Vitamin D3', dosage: '1000 IU', frequency: 'Once daily', nextDose: '8:00 AM', status: 'active' },
  ]

  return (
    <div className="space-y-6">
      <Helmet>
        <title>Medications - Smart Shoe Monitor</title>
        <meta name="description" content="Manage your medications and dosing schedule" />
      </Helmet>

      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
        <div className="flex justify-between items-center">
          <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Medications</h1>
            <p className="mt-2 text-gray-600 dark:text-gray-400">Manage your medication schedule and reminders</p>
          </div>
          <Button className="flex items-center gap-2">
            <Plus className="w-4 h-4" />
            Add Medication
          </Button>
        </div>
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {mockMedications.map((med, index) => (
          <Card key={med.id} className="p-6">
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                  <Pill className="w-5 h-5 text-green-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900 dark:text-white">{med.name}</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">{med.dosage}</p>
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <div className="flex items-center gap-2 text-sm">
                <Clock className="w-4 h-4 text-gray-500" />
                <span className="text-gray-600 dark:text-gray-400">{med.frequency}</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <AlertTriangle className="w-4 h-4 text-orange-500" />
                <span className="text-gray-600 dark:text-gray-400">Next: {med.nextDose}</span>
              </div>
            </div>

            <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
              <div className="flex gap-2">
                <Button variant="outline" size="sm" className="flex-1">
                  Edit
                </Button>
                <Button variant="outline" size="sm" className="flex-1">
                  Take Now
                </Button>
              </div>
            </div>
          </Card>
        ))}
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
        <Card title="Medication Reminders">
          <div className="text-center py-8">
            <Clock className="w-12 h-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">No upcoming reminders</h3>
            <p className="text-gray-600 dark:text-gray-400">You're all caught up with your medications!</p>
          </div>
        </Card>
      </motion.div>
    </div>
  )
}

export default Medications