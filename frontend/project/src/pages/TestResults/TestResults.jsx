import { useState } from 'react'
import Card from '../../components/common/Card'
import Button from '../../components/common/Button'
import TestChart from './components/TestChart'
import ResultSummary from './components/ResultSummary'

function TestResults() {
  const [selectedTest, setSelectedTest] = useState('vibration')
  
  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-neutral-900">Test Results</h1>
          <p className="mt-1 text-neutral-500">
            View and analyze patient test results and historical data
          </p>
        </div>
        <div className="mt-4 md:mt-0 flex space-x-2">
          <Button 
            variant="outline" 
            leftIcon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
              </svg>
            }
          >
            Filter Results
          </Button>
          <Button
            leftIcon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            }
          >
            Export Data
          </Button>
        </div>
      </div>
      
      {/* Test type selector */}
      <div className="bg-white shadow-card rounded-lg p-4 border border-neutral-200">
        <div className="flex flex-wrap gap-2">
          <Button 
            variant={selectedTest === 'vibration' ? 'primary' : 'outline'}
            onClick={() => setSelectedTest('vibration')}
          >
            Vibration Test
          </Button>
          <Button 
            variant={selectedTest === 'pressure' ? 'primary' : 'outline'}
            onClick={() => setSelectedTest('pressure')}
          >
            Pressure Test
          </Button>
          <Button 
            variant={selectedTest === 'temperature' ? 'primary' : 'outline'}
            onClick={() => setSelectedTest('temperature')}
          >
            Temperature Test
          </Button>
        </div>
      </div>
      
      {/* Test results content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Test visualization */}
        <div className="lg:col-span-2">
          <Card title={`${selectedTest.charAt(0).toUpperCase() + selectedTest.slice(1)} Test Results`}>
            <TestChart testType={selectedTest} />
          </Card>
        </div>
        
        {/* Results summary */}
        <div>
          <ResultSummary testType={selectedTest} />
        </div>
      </div>
    </div>
  )
}

export default TestResults