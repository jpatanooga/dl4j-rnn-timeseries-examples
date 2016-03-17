package org.deeplearning4j.examples.rnn.strata.physionet.utils;

import java.util.ArrayList;
import java.util.List;

public class EvalScoreTracker {
	
	List<Integer> epochs = new ArrayList<>();
	List<Double> f1Scores_train = new ArrayList<>();
	List<Double> f1Scores_validate = new ArrayList<>();
	int windowSize = 5;
	
	public EvalScoreTracker(int windowSize) {
		
		this.windowSize = windowSize;
		
		
	}
	
	public void addF1(int epoch, double f1_train, double f1_validate) {
		
		if (this.f1Scores_train.size() >= this.windowSize) {
			this.f1Scores_train.remove(0);
		}

		if (this.f1Scores_validate.size() >= this.windowSize) {
			this.f1Scores_validate.remove(0);
		}
		
		if (this.epochs.size() >= this.windowSize) {
			this.epochs.remove(0);
		}
		
		this.f1Scores_train.add( new Double( f1_train ) );
		this.f1Scores_validate.add( new Double( f1_validate ) );
		this.epochs.add( new Integer( epoch ) );
		
	}
	
	public void printWindow() {
		
		System.out.println( "> ---------------- ---------------- ----------------" );
		System.out.println( "Last " + this.windowSize + " f1 Scores: " );
		
		for (int x = 0; x < this.f1Scores_train.size(); x++ ) {
			
			
			System.out.println( "\tEpoch: " + this.epochs.get( x ) + ", F1-Train: " + this.f1Scores_train.get(x) + ", F1-Validate: " + this.f1Scores_validate.get(x) );
			
		}
		
		System.out.println( "> ---------------- ---------------- ----------------" );
		
	}

}
