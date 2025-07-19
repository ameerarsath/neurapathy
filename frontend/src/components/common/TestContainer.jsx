import React from 'react';
import { useRole } from '../../contexts/RoleContext';
import BaseTest from '../common/BaseTest';
import PhysicianTestView from '../common/PhysicianTestView';

const TestContainer = ({ testType }) => {
  const { userRole } = useRole();

  return userRole === 'physician' ? (
    <PhysicianTestView testType={testType} />
  ) : (
    <BaseTest testType={testType} />
  );
};

export default TestContainer;
