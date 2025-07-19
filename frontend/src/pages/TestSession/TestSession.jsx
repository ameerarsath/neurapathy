import { useState } from 'react'
import { motion } from 'framer-motion'
import { Helmet } from 'react-helmet-async'
import { Play, Pause, Square, Activity, Timer, Target } from 'lucide-react'
import Card from '@components/common/Card'
import Button from '@components/common/Button'

function TestSession() {
  const [isActive, setIsActive] = useState(false)
  const [currentTest, setCurrentTest] = useState('vibration')

  const tests = [
    { id: 'vibration', name: 'Vibration Test', duration: 300 },
    { id: 'pressure', name: 'Pressure Test', duration: 240 },
    { id: 'temperature', name: 'Temperature Test', duration: 180 },
  ]

  return (
    <div className="space-y-6">
      <Helmet>
        <title>Test Session - Smart Shoe Monitor</title>
        <meta name="description" content="Conduct neuropathy assessment tests" />
      </Helmet>

      <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }}>
        <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Test Session</h1>
        <p className="mt-2 text-gray-600 dark:text-gray-400">Conduct your neuropathy assessment</p>
      </motion.div>

      <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Test Control */}
        <div className="lg:col-span-2">
          <Card title="Current Test: Vibration Assessment">
            <div className="text-center py-12">
              <motion.div animate={{ scale: isActive ? [1, 1.1, 1] : 1 }} transition={{ repeat: isActive ? Infinity : 0, duration: 2 }} className="w-24 h-24 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-6">
                <Activity className="w-12 h-12 text-blue-600" />
              </motion.div>
              
              <h3 className="text-xl font-semibold text-gray-900 dark:text-white mb-4">
                {isActive ? 'Test in Progress' : 'Ready to Start'}
              </h3>
              
              <div className="flex justify-center gap-4">
                <Button onClick={() => setIsActive(!isActive)} className="flex items-center gap-2">
                  {isActive ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
                  {isActive ? 'Pause' : 'Start'} Test
                </Button>
                <Button variant="outline" onClick={() => setIsActive(false)}>
                  <Square className="w-4 h-4 mr-2" />
                  Stop
                </Button>
              </div>
            </div>
          </Card>
        </div>

        {/* Test Selection */}
        <div>
          <Card title="Available Tests">
            <div className="space-y-3">
              {tests.map((test) => (
                <button key={test.id} onClick={() => setCurrentTest(test.id)} className={`w-full p-3 rounded-lg border text-left transition-colors ${currentTest === test.id ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20' : 'border-gray-200 hover:border-gray-300'}`}>
                  <div className="font-medium text-gray-900 dark:text-white">{test.name}</div>
                  <div className="text-sm text-gray-600 dark:text-gray-400 flex items-center gap-1">
                    <Timer className="w-3 h-3" />
                    {test.duration}s
                  </div>
                </button>
              ))}
            </div>
          </Card>

          <Card title="Instructions" className="mt-4">
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-2">
              <li>• Ensure device is properly connected</li>
              <li>• Sit comfortably with feet positioned correctly</li>
              <li>• Follow the prompts during the test</li>
              <li>• Remain still during measurements</li>
            </ul>
          </Card>
        </div>
      </motion.div>
    </div>
  )
}

export default TestSession