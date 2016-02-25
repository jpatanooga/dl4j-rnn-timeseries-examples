package org.deeplearning4j.jp.rnn.synthetic.simple3.uneven;

import static org.junit.Assert.*;

import java.io.IOException;

import org.deeplearning4j.examples.rnn.synthetic.ND4JMatrixTool;
import org.deeplearning4j.examples.rnn.synthetic.simple1.SyntheticDataIterator;
import org.deeplearning4j.examples.rnn.synthetic.simple3.uneven.Simple3_Uneven_Dataset_Iterator;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

public class Test_Simple3_Uneven_Dataset_Iterator {

	@Test
	public void test() throws IOException {
		
		Simple3_Uneven_Dataset_Iterator iter = new Simple3_Uneven_Dataset_Iterator("src/test/resources/data/synthetic/simple_3_uneven/simple_3_uneven_data.txt", "src/test/resources/data/synthetic/simple_3_uneven/simple_3_uneven_labels.txt", 40, 40, 4 );
		
		//iter.next();
		
        DataSet t = iter.next();
        INDArray input = t.getFeatureMatrix();
        INDArray labels = t.getLabels();
        INDArray inputMask = t.getFeaturesMaskArray();
        INDArray labelsMask = t.getLabelsMaskArray();
		
        int miniBatchSize = 40;
        int columnCount = 1;
        int timestepCount = 4;
        
		System.out.println("\n\nDebug Input");
		ND4JMatrixTool.debug3D_Nd4J_Input(input, miniBatchSize, columnCount, timestepCount);
		ND4JMatrixTool.debug2D_Nd4J_Input( inputMask, miniBatchSize, timestepCount);
		
		
		System.out.println("\n\nDebug Labels");
		ND4JMatrixTool.debug3D_Nd4J_Input(labels, miniBatchSize, 2, timestepCount);
		ND4JMatrixTool.debug2D_Nd4J_Input( labelsMask, miniBatchSize, timestepCount);
        
        
		
	}

}
