"use client";

import { useState } from "react";
import clsx from "clsx";
import { BookContentView } from "./content-view";
import { BookRecommendationView } from "./recommendation-view";

const GridIcon = (props: any) => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
    <path d="M10 3H3V10H10V3Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M21 3H14V10H21V3Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M21 14H14V21H21V14Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M10 14H3V21H10V14Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

const BoxIcon = (props: any) => (
  <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" {...props}>
    <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
    <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
  </svg>
);

interface DetailWrapperProps {
  book: any;
}

export const BookDetailWrapper = ({ book }: DetailWrapperProps) => {
  const [activeTab, setActiveTab] = useState<"contents" | "recommendation">("contents");

  const menuItems = [
    { key: "contents", label: "Contents", icon: <GridIcon /> },
    { key: "recommendation", label: "Recommendation", icon: <BoxIcon /> },
  ];

  return (
    <div className="flex flex-col md:flex-row min-h-screen w-screen relative left-[50%] right-[50%] -ml-[50vw] -mr-[50vw] bg-background">
            <aside className="hidden md:block w-72 border-r border-default-200 bg-default-50/50 fixed left-0 top-0 h-screen flex-shrink-0 z-40">
        <div className="h-full overflow-y-auto p-6 pt-20">
          <div className="mb-8 px-2">
             <h2 className="text-xl font-bold text-foreground">Detail Buku</h2>
             <p className="text-tiny text-default-500">ID: {book.id}</p>
          </div>

          <ul className="flex flex-col gap-2">
            {menuItems.map((item) => (
              <li key={item.key}>
                <button
                  onClick={() => setActiveTab(item.key as any)}
                  className={clsx(
                    "flex items-center gap-3 w-full px-4 py-3 rounded-xl transition-all duration-200 text-sm font-medium",
                    activeTab === item.key
                      ? "bg-default-900 text-default-50 dark:bg-default-100 dark:text-default-900 shadow-md font-bold" 
                      : "text-default-500 hover:text-default-900 hover:bg-default-100"
                  )}
                >
                  {item.icon}
                  {item.label}
                </button>
              </li>
            ))}
          </ul>
        </div>
      </aside>

      <main className="flex-grow w-full md:ml-72"> 
        <div className="w-full p-6 md:p-12">
          
          <div className="md:hidden flex gap-2 mb-6 overflow-x-auto pb-2 border-b border-default-100">
             {menuItems.map((item) => (
                <button
                  key={item.key}
                  onClick={() => setActiveTab(item.key as any)}
                  className={clsx(
                    "px-4 py-2 rounded-full text-sm whitespace-nowrap border",
                    activeTab === item.key 
                        ? "bg-foreground text-background border-foreground font-bold"
                        : "bg-transparent text-default-500 border-default-200"
                  )}
                >
                   {item.label}
                </button>
             ))}
          </div>

          {activeTab === "contents" ? (
            <BookContentView book={book} />
          ) : (
            <BookRecommendationView currentBookId={book.id} />
          )}
        </div>
      </main>

    </div>
  );
};