import React from "react";

export const Card = ({ children, className = "" }) => (
  <div className={`bg-white shadow rounded p-4 ${className}`}>{children}</div>
);

export const CardHeader = ({ children }) => (
  <div className="border-b mb-4">{children}</div>
);

export const CardTitle = ({ children }) => (
  <h2 className="text-lg font-bold">{children}</h2>
);

export const CardContent = ({ children }) => <div>{children}</div>;
