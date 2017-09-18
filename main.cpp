//  @author    Mohammad Mohebi
//  @file      main.cpp
//

#include "Matrix.hpp"

#include <cstdlib>
#include <ctime>
#include <iostream>
#include <stdexcept>
#include <mpi.h>
#include "Inverter.hpp"

using namespace std;

// Inverser la matrice par la méthode de Gauss-Jordan; implantation séquentielle.
void invertSequential(Matrix& iA) {
    
    // vérifier que la matrice est carrée
    assert(iA.rows() == iA.cols());
	// construire la matrice [A I]
	MatrixConcatCols lAI(iA, MatrixIdentity(iA.rows()));
    
    // traiter chaque rangée
	for (size_t k=0; k<iA.rows(); ++k) {
		// trouver l'index p du plus grand pivot de la colonne k en valeur absolue
		// (pour une meilleure stabilité numérique).
		size_t p = k;
        double lMax = fabs(lAI(k,k));
        for(size_t i = k; i < lAI.rows(); ++i) {
            if(fabs(lAI(i,k)) > lMax) {
                lMax = fabs(lAI(i,k));
                p = i;
            }
        }
        // vérifier que la matrice n'est pas singulière
		if (lAI(p, k) == 0) throw runtime_error("Matrix not invertible");
        
        // échanger la ligne courante avec celle du pivot
		if (p != k) lAI.swapRows(p, k);
        
		double lValue = lAI(k, k);
		for (size_t j=0; j<lAI.cols(); ++j) {
			// On divise les éléments de la rangée k
			// par la valeur du pivot.
			// Ainsi, lAI(k,k) deviendra égal à 1.
			lAI(k, j) /= lValue;
		}
        
		// Pour chaque rangée...
		for (size_t i=0; i<lAI.rows(); ++i) {
			if (i != k) { // ...différente de k
                // On soustrait la rangée k
                // multipliée par l'élément k de la rangée courante
				double lValue = lAI(i, k);
                lAI.getRowSlice(i) -= lAI.getRowCopy(k)*lValue;
			}
		}
	}
	
	// On copie la partie droite de la matrice AI ainsi transformée
	// dans la matrice courante (this).
	for (unsigned int i=0; i<iA.rows(); ++i) {
		iA.getRowSlice(i) = lAI.getDataArray()[slice(i*lAI.cols()+iA.cols(), iA.cols(), 1)];
	}
}

// Inverser la matrice par la méthode de Gauss-Jordan; implantation MPI parallèle.
void invertParallel(Matrix& iA) {
	int lSize = MPI::COMM_WORLD.Get_size();
	int lRank = MPI::COMM_WORLD.Get_rank();
	
	if (lRank == 0) {
		cout << "Dimension du monde = " << lSize << endl;
	}
	
	Inverter invertMatrix(iA);
	invertMatrix.setRank(lRank);
	invertMatrix.process();
	
	cout << "Bonjour le monde, je suis le processus " << lRank << endl;
}

// Multiplier deux matrices.
Matrix multiplyMatrix(const Matrix& iMat1, const Matrix& iMat2) {
    
    // vérifier la compatibilité des matrices
    assert(iMat1.cols() == iMat2.rows());
    // effectuer le produit matriciel
    Matrix lRes(iMat1.rows(), iMat2.cols());
    // traiter chaque rangée
    for(size_t i=0; i < lRes.rows(); ++i) {
        // traiter chaque colonne
        for(size_t j=0; j < lRes.cols(); ++j) {
            lRes(i,j) = (iMat1.getRowCopy(i)*iMat2.getColumnCopy(j)).sum();
        }
    }
    return lRes;
}

int main(int argc, char** argv) {
    
	MPI::Init();
	srand((unsigned)time(NULL));
	int lRank = MPI::COMM_WORLD.Get_rank();
    
	unsigned int lS = 5;
	uint uiCalcType = 0;
	if (argc >= 2) {
		lS = atoi(argv[1]);
	}
    
	Matrix lA(lS, lS);
	Matrix lB(lS, lS);
	if(lRank == 0){
		MatrixRandom rand(lS, lS);
		cout << "Matrice random:\n" << rand.str() << endl;
		lA = rand;
		lB = rand;
		cout << "Matrice random:\n" << lA.str() << endl;
    } 
	
	if(uiCalcType == 0)
	{
		if(lRank == 0)
			cout << endl << "------------------------- Inversion parallèle -------------------------" << endl;
		invertParallel(lB);	
	}
	else
	{		
		if(lRank == 0)
		{
			cout << endl << "------------------------- Inversion sequentiel ---------------------------" << endl;
			invertSequential(lB);
		}
	}
	if(lRank == 0)
		cout << "Matrice inverse:\n" << lB.str() << endl;
    
	if(lRank == 0){
		Matrix lRes = multiplyMatrix(lA, lB);
		cout << "Produit des deux matrices:\n" << lRes.str() << endl;
    
		cout << "Erreur: " << lRes.getDataArray().sum() - lS << endl;
	}
	MPI::Finalize();
	return 0;
}

