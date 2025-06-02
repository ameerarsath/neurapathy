import Card from '../../../components/common/Card'
import Button from '../../../components/common/Button'

function ResultSummary({ testType }) {
  // Mock data for test results - in a real app, this would come from an API
  const getTestData = () => {
    switch (testType) {
      case 'vibration':
        return {
          title: 'Vibration Test',
          score: 68,
          change: '-3%',
          changeType: 'negative',
          lastTest: '2023-12-10',
          details: [
            { label: 'Left Foot Score', value: '72/100' },
            { label: 'Right Foot Score', value: '64/100' },
            { label: 'Sensation Threshold', value: 'Moderately Reduced' },
            { label: 'Progression', value: 'Slow Decline' },
          ],
          recommendations: [
            'Continue prescribed physical therapy exercises',
            'Increase foot examination frequency to twice weekly',
            'Consider consultation with neurologist for further evaluation',
          ]
        }
      case 'pressure':
        return {
          title: 'Pressure Points Test',
          score: 73,
          change: '-1%',
          changeType: 'neutral',
          lastTest: '2023-12-10',
          details: [
            { label: 'High Pressure Areas', value: '3 Points' },
            { label: 'Right Heel', value: 'Elevated (27%)' },
            { label: 'Left Forefoot', value: 'Elevated (22%)' },
            { label: 'Weight Distribution', value: 'Uneven' },
          ],
          recommendations: [
            'Continue using prescribed orthotic inserts',
            'Consider adjusting footwear to reduce pressure points',
            'Monitor right heel area for potential ulceration',
          ]
        }
      case 'temperature':
        return {
          title: 'Temperature Mapping',
          score: 82,
          change: '+2%',
          changeType: 'positive',
          lastTest: '2023-12-10',
          details: [
            { label: 'Average Temperature', value: '32.5°C' },
            { label: 'Temperature Variance', value: '2.7°C' },
            { label: 'Cold Spots', value: '2 Areas (Toes)' },
            { label: 'Circulation Status', value: 'Moderately Reduced' },
          ],
          recommendations: [
            'Continue with circulation-improving exercises',
            'Keep feet warm with appropriate socks',
            'Avoid exposure to cold temperatures',
          ]
        }
      default:
        return {}
    }
  }
  
  const testData = getTestData()
  
  // Determine score color
  const getScoreColor = (score) => {
    if (score >= 80) return 'text-success-600'
    if (score >= 60) return 'text-warning-600'
    return 'text-error-600'
  }
  
  // Determine change color
  const getChangeColor = (changeType) => {
    switch (changeType) {
      case 'positive':
        return 'text-success-600'
      case 'negative':
        return 'text-error-600'
      default:
        return 'text-neutral-600'
    }
  }
  
  return (
    <div className="space-y-4">
      <Card title="Test Summary">
        <div className="flex flex-col items-center mb-4">
          <div className="relative">
            <svg className="w-24 h-24" viewBox="0 0 100 100">
              {/* Background circle */}
              <circle 
                cx="50" 
                cy="50" 
                r="40" 
                fill="none" 
                stroke="#e5e7eb" 
                strokeWidth="10" 
              />
              
              {/* Progress circle */}
              <circle 
                cx="50" 
                cy="50" 
                r="40" 
                fill="none" 
                stroke={testData.score >= 80 ? '#10B981' : testData.score >= 60 ? '#F59E0B' : '#EF4444'} 
                strokeWidth="10" 
                strokeDasharray="251.2" 
                strokeDashoffset={(251.2 * (100 - testData.score)) / 100} 
                transform="rotate(-90 50 50)" 
              />
            </svg>
            <div className="absolute inset-0 flex items-center justify-center">
              <span className={`text-2xl font-bold ${getScoreColor(testData.score)}`}>{testData.score}</span>
            </div>
          </div>
          
          <div className="mt-2 text-sm text-neutral-600">
            <span>Last test: {testData.lastTest}</span>
            <span className={`ml-2 ${getChangeColor(testData.changeType)}`}>
              {testData.change}
            </span>
          </div>
        </div>
        
        <div className="space-y-4">
          <div>
            <h4 className="text-sm font-medium text-neutral-700 mb-2">Details</h4>
            <div className="space-y-2">
              {testData.details?.map((detail, index) => (
                <div key={index} className="flex justify-between">
                  <span className="text-sm text-neutral-600">{detail.label}</span>
                  <span className="text-sm font-medium">{detail.value}</span>
                </div>
              ))}
            </div>
          </div>
          
          <div>
            <h4 className="text-sm font-medium text-neutral-700 mb-2">Recommendations</h4>
            <ul className="list-disc pl-5 space-y-1">
              {testData.recommendations?.map((rec, index) => (
                <li key={index} className="text-sm text-neutral-600">{rec}</li>
              ))}
            </ul>
          </div>
        </div>
      </Card>
      
      <Card title="Recent Tests">
        <div className="space-y-3">
          <div className="p-3 border border-neutral-200 rounded-md">
            <div className="flex justify-between mb-1">
              <span className="font-medium">{testData.title}</span>
              <span className="text-sm text-neutral-500">Dec 10, 2023</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-neutral-600">Score: {testData.score}/100</span>
              <Button variant="ghost" size="sm" className="text-xs py-0 px-2">View</Button>
            </div>
          </div>
          
          <div className="p-3 border border-neutral-200 rounded-md">
            <div className="flex justify-between mb-1">
              <span className="font-medium">{testData.title}</span>
              <span className="text-sm text-neutral-500">Nov 25, 2023</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-neutral-600">Score: {testData.score - 3}/100</span>
              <Button variant="ghost" size="sm" className="text-xs py-0 px-2">View</Button>
            </div>
          </div>
          
          <div className="p-3 border border-neutral-200 rounded-md">
            <div className="flex justify-between mb-1">
              <span className="font-medium">{testData.title}</span>
              <span className="text-sm text-neutral-500">Nov 11, 2023</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-neutral-600">Score: {testData.score - 5}/100</span>
              <Button variant="ghost" size="sm" className="text-xs py-0 px-2">View</Button>
            </div>
          </div>
        </div>
        
        <div className="mt-4 text-center">
          <Button variant="outline" size="sm">View All Tests</Button>
        </div>
      </Card>
    </div>
  )
}

export default ResultSummary