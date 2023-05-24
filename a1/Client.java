import java.rmi.*;
import java.util.Scanner;

public class Client{
	public static void main(String[] args){
		
		Scanner sc = new Scanner(System.in);
		
		try{
			String serverUrl = "rmi://localhost/Server";
			ServerInterface serverIntf = (ServerInterface) Naming.lookup(serverUrl);
			
			System.out.print("Enter Fisrt Number: ");
			double num1 = sc.nextDouble();
			
			System.out.print("Enter Second Number: ");
			double num2 = sc.nextDouble();
			
			System.out.println("*--------------------------Results----------------------------*");
			System.out.println("Addition is: " + serverIntf.Addition(num1,num2));
			System.out.println("Subtraction is: " + serverIntf.Subtraction(num1,num2));
			System.out.println("Division is: " + serverIntf.Division(num1,num2));
			System.out.println("Multiplication is: " + serverIntf.Multiplication(num1,num2));
		}
		
		catch(Exception e){
			System.out.println("Exception occured at Client! " + e.getMessage());
		}
	}
}
