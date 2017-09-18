//  @author    Mohammad Mohebi
//  @file      Inverter.cpp
//
#include "Inverter.hpp"


Inverter::Inverter(Matrix& matrix)
	:m_matMyMatrix(matrix), m_matPivot(1, matrix.cols())
{
	m_steps = STEP_a_DET_q;
	m_szColToProcess = 0;
	m_dLocalMaxValue = 0;
	m_bContinueProcess = true;
	m_szMaxLocalRow = 0;
}

Inverter::~Inverter()
{
}

void Inverter::process()
{
	MAX_LOC localMax, global;
	int size;
	std::valarray<double> row(1, m_matMyMatrix.cols());
	while(m_bContinueProcess)
	{
		switch(m_steps)
		{
			// Déterminer localement le q parmi les lignes i = k qui appartiennent à r, puis
			// faire une réduction (Allreduce avec MAXLOC) pour déterminer le q global
			case STEP_a_DET_q:
				if(!m_vecRowsIndex.empty())
					m_dLocalMaxValue = abs(m_vecRowsIndex[0]);
				for(size_t i = 0; i < m_vecRowsIndex.size(); i++)
				{
					double tempValue = abs(m_matMyMatrix(m_vecRowsIndex[i], m_szColToProcess));
					if(m_dLocalMaxValue < tempValue)
					{
						m_dLocalMaxValue = tempValue;
						m_szMaxLocalRow = m_vecRowsIndex[i];
					}
				}
				localMax.value = m_dLocalMaxValue;
				localMax.index = m_iMyRank;
				std::cout << "Process: ---- " << m_iMyRank << " ---- Atteint le barrier" << std::endl;
				MPI::COMM_WORLD.Barrier();
				
				std::cout << "Process: ---- " << m_iMyRank << " ---- Allreduce " << std::endl;
				MPI::COMM_WORLD.Allreduce(&localMax, &global, 1, MPI::DOUBLE_INT, MPI::MAXLOC);
				size = sizeof(row)/sizeof(MPI::DOUBLE);
				if(global.index == m_iMyRank)
				{
					row = m_matMyMatrix.getRowCopy(m_szMaxLocalRow);
					//std::cout << row;
				}
				std::cout << "Process: ------ " << m_iMyRank << " ----- Bcast()" << std::endl;
				MPI::COMM_WORLD.Bcast(&row, size, MPI::DOUBLE, global.index);
				
				//std::cout << row;
				// Pou cette étape on swappera pas, c'est à la fin que le processus Rank 0 qui ferra les swap
				break;
				
			// Si la valeur du max est nulle, la matrice est singulière (ne peut être inversée)
			case STEP_b_IS_singuliere:
				break;
			
			// Diffuser (Bcast) la ligne q appartenant au processus r=q%p (r est root)
			case STEP_c_BCAST_q:
				break;
				
			// Transmettre au processus r la ligne k appartenant au processus k%p
			case STEP_d_TRANS_k:
				break;
				
			// Permuter localement les lignes q et k
			case STEP_e_PERMUTE_qk:
				break;
				
			// Normaliser la ligne k afin que l’élément (k,k) égale 1
			case STEP_f_NORMALISE_k:
				break;
				
			// Éliminer les éléments (i,k) pour toutes les lignes i qui appartiennent au
			// processus r, sauf pour la ligne k
			case STEP_g_ELEMIN_ik:
				break;
			
			// Rapatrier (Gatherv) toutes les lignes sur le processus 0
			case STEP_h_GATHER_all:
				break;
			
			// Verifier s'il y a une signal de terminaison à cause d'un erreur quelconque
			case STEP_i_VERIFY_terminate:
				break;
				
			default:
				break;
		}
		MPI::COMM_WORLD.Barrier();
		break;
	}
}

void Inverter::addRowIndex(size_t index)
{
	m_vecRowsIndex.push_back(index);
}

