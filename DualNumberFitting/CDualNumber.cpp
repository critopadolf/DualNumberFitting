#include "CDualNumber.h"
/*
  Source:
  https://blog.demofox.org/2017/02/20/multivariable-dual-numbers-automatic-differentiation/
*/

CDualNumber::CDualNumber()
{
    NUMVARIABLES = 0;
    m_real = 0;
}
// constructor to make a constant
CDualNumber::CDualNumber(int numVars, float f) {
    NUMVARIABLES = numVars;
    m_dual = std::vector<float>(NUMVARIABLES);
    m_real = f;
    std::fill(m_dual.begin(), m_dual.end(), 0.0f);
}

// constructor to make a variable value.  It sets the derivative to 1.0 for whichever variable this is a value for.
CDualNumber::CDualNumber(int numVars, float f, size_t variableIndex) {
    NUMVARIABLES = numVars;
    m_dual = std::vector<float>(NUMVARIABLES);
    m_real = f;
    std::fill(m_dual.begin(), m_dual.end(), 0.0f);
    m_dual[variableIndex] = 1.0f;
}
