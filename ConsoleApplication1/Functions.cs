using System;

namespace Functions
{
    enum FunctionType
    {
        Sigmoid,
        Tangential,
        Gaussian
    }
    delegate double FunctionDel(double x);
    abstract class ActivationFunction
    {
        abstract public FunctionDel GetFunction();
        abstract public FunctionDel GetDerivative();
        abstract public FunctionType GetFunctionType();

        static public ActivationFunction GetFunction(FunctionType type)
        {
            switch (type)
            {
                case FunctionType.Sigmoid:
                    return new SigmoidFunction();
                case FunctionType.Tangential:
                    return new TangentialFunction();
                case FunctionType.Gaussian:
                    return new GaussianFunction();
            }
            throw new Exception("Unknown function type");
        }
    }
    class GaussianFunction : ActivationFunction
    {
        public override FunctionDel GetFunction()
        {
            return new FunctionDel((double x) =>
            (Math.Exp(x * x / (-2)) / (Math.Sqrt(Math.PI * 2))));
        }
        public override FunctionDel GetDerivative()
        {
            return new FunctionDel((double x) =>
            {
                FunctionDel f = GetFunction();
                return -x * f(x);
            });
        }
        public override FunctionType GetFunctionType()
        {
            return FunctionType.Gaussian;
        }
    }
    class SigmoidFunction : ActivationFunction
    {
        override public FunctionDel GetFunction()
        {
            return new FunctionDel((double x) => (1.0 / (1.0 + Math.Exp(-x))));
        }
        override public FunctionDel GetDerivative()
        {
            FunctionDel f = GetFunction();
            return new FunctionDel((double x) =>
            {
                double fx = f(x);
                return fx * (1.0 - fx);
            });
        }
        public override FunctionType GetFunctionType()
        {
            return FunctionType.Sigmoid;
        }
    }
    class TangentialFunction : ActivationFunction
    {
        override public FunctionDel GetFunction()
        {
            return new FunctionDel((double x) => (Math.Tanh(x)));
        }
        override public FunctionDel GetDerivative()
        {
            FunctionDel f = GetFunction();
            return new FunctionDel((double x) =>
            {
                double fx = f(x);
                return 1.0 - fx * fx;
            });
        }
        public override FunctionType GetFunctionType()
        {
            return FunctionType.Tangential;
        }
    }
}