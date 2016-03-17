package org.deeplearning4j.examples.rnn.strata.physionet;

import static org.junit.Assert.assertEquals;

import java.io.File;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.util.Arrays;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.NoSuchElementException;
import java.util.Random;

import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;

public class PhysioNet_ICU_Mortality_Iterator implements DataSetIterator {
	//private static final long serialVersionUID = -7287833919126626356L;
	//private static final int MAX_SCAN_LENGTH = 200; 
	//private char[] validCharacters;
	//private Map<Character,Integer> charToIdxMap;
	// private char[] fileCharacters;
	// private int exampleLength;
	
	private int miniBatchSize;
	private int currentFileListIndex = 0;
	private int totalExamples = 0;
	
	//private int numExamplesToFetch;
	//private int examplesSoFar = 0;
	//private Random rng;
	//private final int numCharacters;
	//private final boolean alwaysStartAtNewLine;
	
	String datasetInputPath = "";
	String datasetSchemaPath = "";
	String datasetLabelsPath = "";
	public PhysioNet_Vectorizer vectorizer = null;
	
	public PhysioNet_ICU_Mortality_Iterator(String dataInputPath, String datasetSchemaPath, String datasetLabels, int miniBatchSize, int totalExamples ) throws IOException {
	//	this(path,Charset.defaultCharset(),miniBatchSize,exampleSize,numExamplesToFetch,getDefaultCharacterSet(), new Random(),true);
		//this.numCharacters = 0; // fix
		
		this.datasetInputPath = dataInputPath;
		this.datasetSchemaPath = datasetSchemaPath;
		this.datasetLabelsPath = datasetLabels;
		
		this.vectorizer = new PhysioNet_Vectorizer(this.datasetInputPath, this.datasetSchemaPath, this.datasetLabelsPath );
		this.vectorizer.loadSchema();
		this.vectorizer.loadLabels();
		
		this.vectorizer.setSpecialStatisticsFileList("/tmp/set-a/");
	//	this.vectorizer.setupBalancedSubset( totalExamples );
		
		this.vectorizer.collectStatistics();

		this.miniBatchSize = miniBatchSize;
		this.totalExamples = totalExamples;
		
	}
	
	
	public boolean hasNext() {
		return currentFileListIndex + miniBatchSize <= this.totalExamples;
	}

	public DataSet next() {
		return next(miniBatchSize);
	}

	/**
	 * TODO: Cut here ----------------------------------
	 * 
	 * Dimensions of input
	 * 	x: miniBatchSize
	 *  y: every column we want to look at per timestep (basically our traditional vector)
	 *  z: the timestep value
	 * 
	 */
	public DataSet next(int miniBatchSize) {
		
		/*
		
		if( examplesSoFar + miniBatchSize > numExamplesToFetch ) throw new NoSuchElementException();
		//Allocate space:
		INDArray input = null; //Nd4j.zeros(new int[]{num,numCharacters,exampleLength});
		INDArray labels = null; //Nd4j.zeros(new int[]{num,numCharacters,exampleLength});
		
		int maxStartIdx = fileCharacters.length - exampleLength;
		
		//Randomly select a subset of the file. No attempt is made to avoid overlapping subsets
		// of the file in the same minibatch
		for( int i=0; i < miniBatchSize; i++ ){
			int startIdx = (int) (rng.nextDouble()*maxStartIdx);
			int endIdx = startIdx + exampleLength;
			int scanLength = 0;
			
			int currCharIdx = charToIdxMap.get(fileCharacters[startIdx]);	//Current input
			int c=0;
			for( int j=startIdx+1; j<=endIdx; j++, c++ ){
				int nextCharIdx = charToIdxMap.get(fileCharacters[j]);		//Next character to predict
				input.putScalar(new int[]{i,currCharIdx,c}, 1.0);
				labels.putScalar(new int[]{i,nextCharIdx,c}, 1.0);
				currCharIdx = nextCharIdx;
			}
		}
		
		examplesSoFar += miniBatchSize;
		return new DataSet(input,labels);
		*/
		
		//int miniBatchSize = 50;
		int columnCount = 0;
		
		
		columnCount = (this.vectorizer.schema.getTransformedVectorSize() + 1);
		
		
		

		
		
//		vec.schema.debugPrintDatasetStatistics();
		/*
		System.out.println( "Max Timesteps: " + this.vectorizer.maxNumberTimeSteps );
		
		System.out.println( "ND4J Input Size: " );
		System.out.println( "Minibatch: " + miniBatchSize );
		System.out.println( "Column Count: " + columnCount );
		System.out.println( "Timestep Count: " + this.vectorizer.maxNumberTimeSteps );
		*/

		//int currentOffset = 0;
				
	//	for ( int index = 0; index < this.vectorizer.listOfFilesToVectorize.length; index += miniBatchSize) {
			
		//	System.out.println( "\n\n ------------- Mini-batch offset: " + this.currentFileListIndex + " -----------------\n" );
			DataSet d = this.vectorizer.generateNextTimeseriesVectorMiniBatch( miniBatchSize, this.currentFileListIndex, columnCount );
			this.currentFileListIndex += miniBatchSize;
			
	//	}		
		
		
			
		return d;
			
	}

	public int totalExamples() {
		return this.currentFileListIndex;
	}

	public int inputColumns() {
		return this.vectorizer.schema.getTransformedVectorSize() + 1;
	}

	public int totalOutcomes() {
		return 2;
	}

	public void reset() {
		this.currentFileListIndex = 0;
	}

	public int batch() {
		return miniBatchSize;
	}

	public int cursor() {
		return this.currentFileListIndex;
	}

	public int numExamples() {
		return this.totalExamples;
	}

	public void setPreProcessor(DataSetPreProcessor preProcessor) {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public List<String> getLabels() {
		throw new UnsupportedOperationException("Not implemented");
	}

	@Override
	public void remove() {
		throw new UnsupportedOperationException();
	}
}
