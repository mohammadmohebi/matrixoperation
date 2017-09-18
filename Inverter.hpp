//  @author    Mohammad Mohebi
//  @file      Inverter.hpp
//
#ifndef _INVERTER_HPP_
#define _INVERTER_HPP_
#include "Matrix.hpp"
#include <mpi.h>

//#define uint unsigned int

class Inverter
{
public:
	Inverter(Matrix& matrix);
	
	~Inverter();
	
	void process();
	
	void setRank(int iRank){m_iMyRank = iRank;}
	void addRowIndex(size_t index);
	
private:
	
	typedef enum{
		STEP_a_DET_q,
		STEP_b_IS_singuliere,
		STEP_c_BCAST_q,
		STEP_d_TRANS_k,
		STEP_e_PERMUTE_qk,
		STEP_f_NORMALISE_k,
		STEP_g_ELEMIN_ik,
		STEP_h_GATHER_all,
		STEP_i_VERIFY_terminate,
		STEP_QTY
	}ALGO_STEP;
	
	typedef struct MAX_LOC{
		double value; 
		int index;
	}MAX_LOC;
	
	std::vector<std::size_t> m_vecRowsIndex;
	
	std::size_t m_szMaxLocalRow;
	std::size_t m_szColToProcess;
	double m_dLocalMaxValue;
	
	Matrix& m_matMyMatrix;
	
	Matrix m_matPivot;
	
	ALGO_STEP m_steps;
	
	int m_iMyRank;
	
	bool m_bContinueProcess;
};


#endif //_INVERTER_HPP_