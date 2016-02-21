package org.deeplearning4j.jp.rnn.strata.physionet;

import static org.junit.Assert.*;

import java.io.IOException;

import org.deeplearning4j.examples.rnn.strata.physionet.PhysioNet_ICU_Mortality_Iterator;
import org.deeplearning4j.examples.rnn.strata.physionet.PhysioNet_Vectorizer;
import org.deeplearning4j.examples.rnn.strata.physionet.schema.TimeseriesDescriptorSchemaColumn;
import org.deeplearning4j.examples.rnn.strata.physionet.schema.TimeseriesSchemaColumn;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;


public class TestPhysioNet_Iterator {

	@Test
	public void testIterateThroughMiniBatches() {
		
		int miniBatchSize = 50;
		int columnCount = 0;
		
		PhysioNet_Vectorizer vec = new PhysioNet_Vectorizer("/tmp/set-a/", "src/test/resources/physionet_schema.txt", "src/test/resources/data/physionet/sample/set-a-labels/Outcomes-a.txt" );
		vec.loadSchema();
		vec.loadLabels();
		
		columnCount = (vec.schema.getTransformedVectorSize() + 1);
		
		assertEquals( "-1", vec.schema.customValueForMissingValue );
		
		TimeseriesDescriptorSchemaColumn column_icutype = vec.schema.getDescriptorColumnSchemaByName("icutype");
		assertEquals( TimeseriesSchemaColumn.ColumnDescriptorMissingValueStrategy.ZERO, column_icutype.missingValStrategy );
		//assertEquals( "5", column_icutype.customMissingValueReplacementValue );

		TimeseriesSchemaColumn column_platelets = vec.schema.getTimeseriesColumnSchemaByName("alp");
		assertEquals( TimeseriesSchemaColumn.ColumnTimeseriesPaddingStrategyType.PAD_TAIL_WITH_ZEROS, column_platelets.paddingStrategy );
		//assertEquals( "0", column_platelets.customMissingValueReplacementValue );
		
		
		
		
		vec.collectStatistics();
		vec.schema.debugPrintDatasetStatistics();
		
		System.out.println( "Max Timesteps: " + vec.maxNumberTimeSteps );
		
		System.out.println( "ND4J Input Size: " );
		System.out.println( "Minibatch: " + miniBatchSize );
		System.out.println( "Column Count: " + columnCount );
		System.out.println( "Timestep Count: " + vec.maxNumberTimeSteps );
		
	//	INDArray input = Nd4j.zeros(new int[]{ miniBatchSize, columnCount, vec.maxNumberTimeSteps });
		// 1 == mini-batch size
		// 2 == number of classes (0 -> no survive, 1 -> survival)
	//	INDArray labels = Nd4j.zeros(new int[]{ miniBatchSize, 2 });
		
//		vec.extractFileContentsAndVectorize( "src/test/resources/physionet_sample_data.txt", 0, columnCount, vec.maxNumberTimeSteps, input, labels);
		
//		PhysioNet_Vectorizer.debug3D_Nd4J_Input( input, miniBatchSize, columnCount, vec.maxNumberTimeSteps );

		int currentOffset = 0;
		
		
		//for ( int index = 0; index < 200; index += miniBatchSize) {
		for ( int index = 0; index < vec.listOfFilesToVectorize.length; index += miniBatchSize) {
				//vec.listOfFilesToVectorize.length; index += miniBatchSize) {
			
			System.out.println( "\n\n ------------- Mini-batch test: " + index + " -----------------\n" );
			DataSet d = vec.generateNextTimeseriesVectorMiniBatch(miniBatchSize, index, columnCount);
			
		}
		
/*		
		double labelNegativeLabelValue = labels.getDouble(0, 0);
		double labelPositiveLabelValue = labels.getDouble(0, 1);
		assertEquals( 0.0, labelNegativeLabelValue, 0.0 );
		assertEquals( 1.0, labelPositiveLabelValue, 0.0 );		
	*/	
		
		
	}
	
	@Test
	public void testIterator() throws IOException {
		
		
		// "/tmp/set-a/", "src/test/resources/physionet_schema.txt", "src/test/resources/data/physionet/sample/set-a-labels/Outcomes-a.txt"
		PhysioNet_ICU_Mortality_Iterator iterator = new PhysioNet_ICU_Mortality_Iterator( "/tmp/set-a/", "src/test/resources/physionet_schema.txt", "src/test/resources/data/physionet/sample/set-a-labels/Outcomes-a.txt", 50, 200);

		while (iterator.hasNext()) {
			
			DataSet d = iterator.next();
			System.out.println( "> Pulled Dataset ... ");
			
		}
		
		System.out.println( "> [ done ] ");

		
	}
	
	

}
