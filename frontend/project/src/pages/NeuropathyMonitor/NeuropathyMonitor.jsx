import Card from '../../components/common/Card'
import Button from '../../components/common/Button'
import FootMap from './components/FootMap'
import ProgressionChart from './components/ProgressionChart'
import SensitivityHeatmap from './components/SensitivityHeatmap'

function NeuropathyMonitor() {
  return (
    <div className="space-y-6 animate-fade-in">
      {/* Page header */}
      <div className="flex flex-col md:flex-row md:items-center md:justify-between">
        <div>
          <h1 className="text-2xl font-semibold text-neutral-900">Neuropathy Monitor</h1>
          <p className="mt-1 text-neutral-500">
            Track neuropathy progression and foot sensitivity
          </p>
        </div>
        <div className="mt-4 md:mt-0 flex space-x-2">
          <Button 
            variant="outline" 
            leftIcon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M8 7V3m8 4V3m-9 8h10M5 21h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v12a2 2 0 002 2z" />
              </svg>
            }
          >
            History
          </Button>
          <Button
            leftIcon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v3m0 0v3m0-3h3m-3 0H9m12 0a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            }
          >
            New Assessment
          </Button>
        </div>
      </div>
      
      {/* Summary cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card>
          <div className="flex flex-col items-center">
            <div className="w-12 h-12 rounded-full bg-primary-100 flex items-center justify-center mb-3">
              <svg className="w-6 h-6 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-neutral-900">Neuropathy Progression</h3>
            <p className="text-3xl font-bold text-warning-600 my-2">Moderate</p>
            <p className="text-sm text-neutral-500 text-center">
              Gradual progression over past 6 months
            </p>
          </div>
        </Card>
        
        <Card>
          <div className="flex flex-col items-center">
            <div className="w-12 h-12 rounded-full bg-secondary-100 flex items-center justify-center mb-3">
              <svg className="w-6 h-6 text-secondary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-neutral-900">Risk Assessment</h3>
            <p className="text-3xl font-bold text-warning-600 my-2">Medium</p>
            <p className="text-sm text-neutral-500 text-center">
              Based on current symptoms and medical history
            </p>
          </div>
        </Card>
        
        <Card>
          <div className="flex flex-col items-center">
            <div className="w-12 h-12 rounded-full bg-accent-100 flex items-center justify-center mb-3">
              <svg className="w-6 h-6 text-accent-600" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <h3 className="text-lg font-medium text-neutral-900">Next Check-up</h3>
            <p className="text-3xl font-bold text-primary-600 my-2">4 days</p>
            <p className="text-sm text-neutral-500 text-center">
              Scheduled for Jan 15, 2024
            </p>
          </div>
        </Card>
      </div>
      
      {/* Main content */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Foot map visualization */}
        <Card title="Foot Sensitivity Map">
          <FootMap />
        </Card>
        
        {/* Progression chart */}
        <Card title="Neuropathy Progression">
          <ProgressionChart />
        </Card>
      </div>
      
      {/* Sensitivity heatmap */}
      <Card title="Sensitivity Heatmap">
        <SensitivityHeatmap />
      </Card>
      
      {/* Recommendations */}
      <Card title="Recommendations">
        <div className="space-y-4">
          <div className="p-3 bg-primary-50 border border-primary-200 rounded-md">
            <div className="flex">
              <div className="mr-3 text-primary-600">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <h4 className="font-medium text-primary-800">Monitor Right Foot</h4>
                <p className="text-sm text-primary-700 mt-1">
                  Increased attention to the right foot, as it shows more significant neuropathy symptoms. 
                  Daily examination recommended.
                </p>
              </div>
            </div>
          </div>
          
          <div className="p-3 bg-warning-50 border border-warning-200 rounded-md">
            <div className="flex">
              <div className="mr-3 text-warning-600">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                </svg>
              </div>
              <div>
                <h4 className="font-medium text-warning-800">Pressure Relief</h4>
                <p className="text-sm text-warning-700 mt-1">
                  Use custom orthotics to redistribute pressure away from the right heel and 
                  left forefoot areas that show elevated pressure readings.
                </p>
              </div>
            </div>
          </div>
          
          <div className="p-3 bg-success-50 border border-success-200 rounded-md">
            <div className="flex">
              <div className="mr-3 text-success-600">
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <div>
                <h4 className="font-medium text-success-800">Exercise Regimen</h4>
                <p className="text-sm text-success-700 mt-1">
                  Continue with the prescribed exercise program to improve circulation and 
                  maintain foot flexibility. Aim for 15 minutes twice daily.
                </p>
              </div>
            </div>
          </div>
        </div>
      </Card>
    </div>
  )
}

export default NeuropathyMonitor