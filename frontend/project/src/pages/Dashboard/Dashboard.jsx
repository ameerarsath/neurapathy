import { useState } from 'react'
import Card from '../../components/common/Card'
import Button from '../../components/common/Button'
import StatusCard from './components/StatusCard'
import SensorReadings from './components/SensorReadings'
import AlertsPanel from './components/AlertsPanel'

function Dashboard() {
  const [timeRange, setTimeRange] = useState('day')
  const patientOverview = {
    name: 'John Doe',
    age: 65,
    lastCheck: '2023-12-20',
    riskScore: 72,
    riskLevel: 'Medium',
    compliance: 'Good',
    nextAppointment: '2024-01-15'
  }
  
  const statusMetrics = [
    { 
      label: 'Risk Score', 
      value: patientOverview.riskScore, 
      icon: 'chart',
      change: '+2.5%',
      changeType: 'negative',
      info: 'Based on recent test results and compliance'
    },
    { 
      label: 'Compliance', 
      value: patientOverview.compliance, 
      icon: 'check',
      change: 'Stable',
      changeType: 'neutral',
      info: 'Medication and monitoring adherence'
    },
    { 
      label: 'Sensation Score', 
      value: '68%', 
      icon: 'foot',
      change: '-1.2%',
      changeType: 'negative',
      info: 'Based on last peripheral test'
    },
    { 
      label: 'Next Check', 
      value: '4 days', 
      icon: 'calendar',
      info: 'Scheduled on January 15, 2024'
    }
  ]
  
  const alerts = [
    {
      id: 1,
      type: 'warning',
      message: 'Increased pressure detected on right heel',
      timestamp: '2 hours ago'
    },
    {
      id: 2,
      type: 'info',
      message: 'Battery level at 30%, please charge your device',
      timestamp: '5 hours ago'
    },
    {
      id: 3,
      type: 'success',
      message: 'Weekly goal achieved: 7 days of continuous monitoring',
      timestamp: '1 day ago'
    }
  ]
  
  return (
    <div className="space-y-6 animate-fade-in">
      
      <div className="flex flex-col md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-neutral-900">Dashboard</h1>
          <p className="mt-1 text-neutral-500">
            Welcome back, Dr. Faheem. Here's the latest patient overview.
          </p>
        </div>
        <div className="mt-4 md:mt-0 flex space-x-2">
          <Button 
            variant="outline" 
            leftIcon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
              </svg>
            }
          >
            Export Data
          </Button>
          <Button
            leftIcon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            }
          >
            New Report
          </Button>
        </div>
      </div>
      
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {statusMetrics.map((metric, index) => (
          <StatusCard key={index} {...metric} />
        ))}
      </div>
      
      
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card 
          title="Patient Overview" 
          className="lg:col-span-1"
          footer={
            <div className="flex justify-end">
              <Button variant="ghost" size="sm">View Details</Button>
            </div>
          }
        >
          <div className="space-y-4">
            <div className="flex justify-between">
              <span className="text-neutral-500">Name</span>
              <span className="font-medium">{patientOverview.name}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-500">Age</span>
              <span>{patientOverview.age} years</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-500">Last Check</span>
              <span>{patientOverview.lastCheck}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-500">Risk Level</span>
              <span className="font-medium text-warning-600">{patientOverview.riskLevel}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-neutral-500">Next Appointment</span>
              <span>{patientOverview.nextAppointment}</span>
            </div>
          </div>
        </Card>
        <Card title="Sensor Readings" className="lg:col-span-2">
          <div className="mb-4 flex justify-between items-center">
            <div className="space-x-2">
              <Button 
                size="sm" 
                variant={timeRange === 'day' ? 'primary' : 'ghost'}
                onClick={() => setTimeRange('day')}
              >
                Day
              </Button>
              <Button 
                size="sm" 
                variant={timeRange === 'week' ? 'primary' : 'ghost'}
                onClick={() => setTimeRange('week')}
              >
                Week
              </Button>
              <Button 
                size="sm" 
                variant={timeRange === 'month' ? 'primary' : 'ghost'}
                onClick={() => setTimeRange('month')}
              >
                Month
              </Button>
            </div>
            
            <div>
              <Button size="sm" variant="outline">
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
                </svg>
                <span className="ml-1">Filter</span>
              </Button>
            </div>
          </div>
          
          <SensorReadings timeRange={timeRange} />
        </Card>
      </div>
      <Card title="Recent Alerts">
        <AlertsPanel alerts={alerts} />
      </Card>
    </div>
  )
}

export default Dashboard