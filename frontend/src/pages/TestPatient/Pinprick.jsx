import React from 'react';
import TestContainer from '../../components/common/TestContainer';

const Pinprick = () => {
  return <TestContainer testType="Pinprick" />;
};

    if (loading) return <div>Loading...</div>

    if (error) return <div>Error: {error.message}</div>

    return (
        <div>
        
            <h1>Pinprick Test Results</h1>
            <p>Left Foot: {pinprickData.leftFoot}</p>
            <p>Right Foot: {pinprickData.rightFoot}</p>
        </div>
  )
}


export default Pinprick